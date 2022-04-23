import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pathlib

import numpy as np
import pandas as pd
from tensorflow._api.v2 import data
import scipy
from scipy.spatial import distance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from transformers import TFBertModel,TFDistilBertModel

import pdb
import json
import os
import sys
from pprint import pprint as mypp
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm


from nlp_data_handle import *
from inlp_debias import get_rowspace_projection,get_projection_to_intersection_of_nullspaces

class TransformerClassifier(keras.Model):
    '''
    This class will have the main model which will pass through the
    transformer and train 
    '''
    def __init__(self,data_args,model_args):
        super(TransformerClassifier,self).__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.cached_bert_embedding_dict = {}

        #Stage 2: ONly used in  Stage 2
        #Now initializing the layers to be used
        if data_args["stage"]==2:
            if "distil" in data_args["transformer_name"]:
                self.bert_model = TFDistilBertModel.from_pretrained(data_args["transformer_name"])
            else:
                self.bert_model = TFBertModel.from_pretrained(data_args["transformer_name"])
            if model_args["train_bert"]==False:
                for layer in self.bert_model.layers:
                    layer.trainable = False
            
            #Setting the dimension of hidden layer
            self.hlayer_dim = model_args["hlayer_dim"]

        #Initializing the heads for each classifier (Domain Sntiment)
        self.cat_emb_layer_list = []
        self.cat_classifier_list = []
        self.cat_importance_weight_list = []
        self.cat_temb_importance_weight_list = []
        for cat in self.data_args["cat_list"]:
            #Stage 2: This is used in Stage 2, when we just have normal classiifer
            #Stage 1: Even the stage 1 guys are using this for now
            if data_args["stage"]==2 or data_args["stage"]==1:
                #Creating the imporatance paramaters for each classifier
                cat_imp_weight = tf.Variable(
                    tf.random_normal_initializer(mean=0.0,stddev=1.0)(
                                    shape=[1,model_args["bemb_dim"]],
                                    dtype=tf.float32,
                    ),
                    trainable=True
                )
                self.cat_importance_weight_list.append(cat_imp_weight)
                
                #Creating an embedding layer
                cat_emb_layer = layers.Dense(self.hlayer_dim,activation="relu")
                self.cat_emb_layer_list.append(cat_emb_layer)
                

            #Stage 1: THis is used in stage 1 when we are working directly with topic
            # if data_args["stage"]==1:
            #     #Creating the topic embedding weights to be used by classifier
            #     cat_temb_imp_weight = tf.Variable(
            #         tf.random_normal_initializer(mean=0.0,stddev=1.0)(
            #                         shape=[1,1,len(data_args["topic_list"])]
            #         ),
            #         trainable=True
            #     )
            #     self.cat_temb_importance_weight_list.append(cat_temb_imp_weight)


            #Stage 1 and 2: This is being used in both the stages
            #Creating a dense layer
            cat_dense = layers.Dense(2,activation="softmax")
            self.cat_classifier_list.append(cat_dense)
        
        #Initializing the heads for the "interpretable" topics
        self.topic_classifier_list = []
        self.topic_embedding_layer_list = []
        self.topic_importance_weight_list = []
        num_topics=len(self.data_args["topic_list"]) if data_args["stage"]==1 else self.data_args["num_topics"]
        for tidx in range(num_topics):
            #Creating the imporatance paramaters for each classifier
            topic_imp_weight = tf.Variable(
                tf.random_normal_initializer(mean=0.0,stddev=1.0)(
                                shape=[1,model_args["bemb_dim"]],
                                dtype=tf.float32,
                ),
                trainable=True
            )
            self.topic_importance_weight_list.append(topic_imp_weight)

            #Creating the topic embedding layers (will be applied before the final activation)
            # topic_embed_layer = layers.Dense(self.model_args["temb_dim"])
            # self.topic_embedding_layer_list.append(topic_embed_layer)

            #Creating a dense layer
            topic_dense = layers.Dense(self.data_args["per_topic_class"],activation="softmax")
            self.topic_classifier_list.append(topic_dense)



        #Initializing the trackers for sentiment classifier
        self.sent_pred_xentropy_list = [
            keras.metrics.Mean(name="sent{}_pred_x".format(cat))
                for cat in self.data_args["cat_list"]
        ]
        self.sent_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="c{}_valid_acc".format(cat))
                for cat in self.data_args["cat_list"]
        ]
        self.sent_l1_loss_list = [ 
            tf.keras.metrics.Mean(name="c{}_l1_loss".format(cat))
                for cat in self.data_args["cat_list"]
        ]

        #Initilaizing the trackers for topics classifier
        self.topic_pred_xentropy_list =[ 
            keras.metrics.Mean(name="topic{}_pred_x".format(tidx))
                for tidx in range(num_topics)
        ] 
        self.topic_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="t{}_valid_acc".format(topic))
                for topic in range(num_topics)
        ]
        self.topic_l1_loss_list = [ 
            tf.keras.metrics.Mean(name="t{}_l1_loss".format(cat))
                for cat in range(num_topics)
        ]
    
    def reset_all_metrics(self):
        for tidx in range(len(self.data_args["topic_list"])):
            self.topic_valid_acc_list[tidx].reset_states()
            self.topic_pred_xentropy_list[tidx].reset_states()
        for cidx in range(len(self.data_args["cat_list"])):
            self.sent_valid_acc_list[cidx].reset_states()
            self.sent_pred_xentropy_list[cidx].reset_states()

    def compile(self, optimizer):
        super(TransformerClassifier, self).compile()
        self.optimizer = optimizer 
    
    def get_sentiment_pred_prob(self,bidx,purpose,idx_train,mask_train,gate_tensor,cidx):
        '''
        bidx    : the batch idx for the dataset
        purpose : train or for validation
        '''
        if self.model_args["cached_bemb"]==False or ((bidx,purpose) not in self.cached_bert_embedding_dict):    
            #Getting the bert activation
            bert_outputs=self.bert_model(
                        input_ids=idx_train,
                        attention_mask=mask_train
            )
            bert_seq_output = bert_outputs.last_hidden_state

            #Multiplying this embedding with corresponding weights
            cat_imp_weights = tf.sigmoid(self.cat_importance_weight_list[cidx])
            weighted_bert_seq_output = bert_seq_output * cat_imp_weights

            #Now we need to take average of the last hidden state:
            #Dont use function, the output shape is not defined
            m = tf.cast(mask_train,dtype=tf.float32)

            masked_sum =tf.reduce_sum(
                            weighted_bert_seq_output * tf.expand_dims(m,-1),
                            axis=1
            )

            num_tokens = tf.reduce_sum(m,axis=1,keepdims=True)

            avg_embedding = masked_sum / (num_tokens+1e-10)

            #Caching the bert embedding for this bidx
            if self.model_args["cached_bemb"]==True:
                self.cached_bert_embedding_dict[(bidx,purpose)] = avg_embedding
        else:
            assert self.model_args["train_bert"]==False,"Caching the trainable bert embedding"
            avg_embedding = self.cached_bert_embedding_dict[(bidx,purpose)]

        #Now we will gate the dimension which are spurious
        # gated_avg_embedding = self.cat_emb_layer_list[cidx](avg_embedding)#*gate_tensor
        
        #Now we will apply the dense layer for this category
        cat_class_prob = self.cat_classifier_list[cidx](avg_embedding)

        return cat_class_prob,avg_embedding
    
    def get_main_pred_prob_final(self,X_enc,cidx):
        '''
        Given the encoded (and projected) sentence embedding we will pass this
        through the final main aka sentiment task classifier.
        '''
        X_prob = self.cat_classifier_list[cidx](X_enc)
        return X_prob 
    
    def get_topic_pred_prob_final(self,X_enc,tidx):
        '''
        '''
        X_prob = self.topic_classifier_list[tidx](X_enc)
        return X_prob
 
    def get_topic_pred_prob_from_sentiment_emb(self,sentiment_emb,tidx,reverse_grad):
        '''
        This will be mainly used in the stage2, when we will try to remove the information
        about the topic from the senitment embedding used by the classifier to see if they 
        are using the topic information.
        '''
        #Reverse the gradient from this sentiment embessing
        # if reverse_grad==True:
        #     sentiment_emb = grad_reverse(sentiment_emb)

        #Predcit the topic information fro mthe given sentiment embedding
        topic_class_prob = self.topic_classifier_list[tidx](sentiment_emb)
        return topic_class_prob

    def get_sentiment_pred_prob_topic_basis(self,idx_train,mask_train,gate_tensor,cidx):
        '''
        Here we will try to get the task prediciton using the topic basis
        i.e the embedding in the topic space to do the sentiment classifiecation
        '''
        #Getting the bert representation
        bert_seq_output = self.get_bert_representation(idx_train,mask_train)

        #Now we will get the topic embedding one by one and weight them
        topic_embedding_list = []
        for tidx in range(len(self.data_args["topic_list"])):
            #Getting the topic_embedding
            _,topic_emb = self.get_topic_pred_prob(bert_seq_output,mask_train,gate_tensor,tidx)
            topic_embedding_list.append(topic_emb)

        #Concatenating all of them in one place
        all_topic_embedding = tf.stack(topic_embedding_list,axis=-1)

        #Geting the avg topic embedding
        cat_temb_imp_weight = tf.sigmoid(self.cat_temb_importance_weight_list[cidx])
        avg_topic_embedding = all_topic_embedding*cat_temb_imp_weight
        avg_topic_embedding = tf.reduce_mean(avg_topic_embedding,axis=-1)

        #Now we will apply our cat classifier
        cat_class_prob = self.cat_classifier_list[cidx](avg_topic_embedding)

        return cat_class_prob
    
    def get_sentiment_pred_prob_topic_direct(self,topic_feature,cidx):
        #Simply we will apply the cat classifier on top of this
        weighted_feature = self.cat_importance_weight_list[cidx]*topic_feature
        pred_prob = self.cat_classifier_list[cidx](weighted_feature)

        return pred_prob

    def get_bert_representation(self,idx_train,mask_train):
        '''
        '''
        bert_outputs=self.bert_model(
                    input_ids=idx_train,
                    attention_mask=mask_train
        )
        bert_seq_output = bert_outputs.last_hidden_state

        return bert_seq_output

    def get_topic_pred_prob(self,bert_seq_output,mask_train,gate_tensor,tidx): 
        # raise NotImplementedError   
        # #Getting the bert activation
        # bert_seq_output = self.get_bert_representation(
        #                             idx_train=idx_train,
        #                             mask_train=mask_train,
        # )
        
        #Stopping the gradient update in case of bert training
        bert_seq_output = tf.stop_gradient(bert_seq_output)

        #Multiplying this embedding with corresponding weights
        topic_imp_weights = tf.sigmoid(self.topic_importance_weight_list[tidx])
        weighted_bert_seq_output = bert_seq_output * topic_imp_weights

        #Now we need to take average of the last hidden state:
        #Dont use function, the output shape is not defined
        m = tf.cast(mask_train,dtype=tf.float32)

        masked_sum =tf.reduce_sum(
                        weighted_bert_seq_output * tf.expand_dims(m,-1),
                        axis=1
        )

        num_tokens = tf.reduce_sum(m,axis=1,keepdims=True)

        avg_embedding = masked_sum / (num_tokens+1e-10)

        #Now we will gate the dimension which are spurious
        gated_avg_embedding = avg_embedding*gate_tensor

        #Applying the hidden layer on this BERT embedding
        topic_embedding = self.topic_embedding_layer_list[tidx](gated_avg_embedding)
        #Next we will normalize the embedding to the norm
        if(self.model_args["normalize_temb"]==True):
            topic_embedding = tf.math.l2_normalize(topic_embedding,axis=-1)
        
        #Now we will apply the dense layer for this topic
        topic_class_prob = self.topic_classifier_list[tidx](topic_embedding)

        return topic_class_prob,topic_embedding
    
    def train_step_stage1(self,sidx,single_ds,gate_tensor,task):
        '''
        This function will run one step of training for the classifier
        '''
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # cxentropy_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

        # #Get the dataset for each category
        # cat_dataset_list = data

        #Keeping track of classification accuracy
        cat_accuracy_list = []

        #Now getting the classification loss one by one each category        
        # cat_total_loss = 0.0
        # for cidx,cat in enumerate(self.data_args["cat_list"]):
        label = single_ds["label"]
        idx = single_ds["topic_feature"]
        # mask = single_ds["attn_mask"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the train data
        label_train = label[0:valid_idx]
        idx_train = idx[0:valid_idx]
        # mask_train = mask[0:valid_idx]

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        # mask_valid = mask[valid_idx:]

        if task=="sentiment":
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape() as tape:
                #Forward propagating the model
                # train_prob = self.get_sentiment_pred_prob_topic_basis(idx_train,mask_train,gate_tensor,sidx)
                train_prob = self.get_sentiment_pred_prob_topic_direct(idx_train,sidx)

                #Getting the loss for this classifier
                xentropy_loss = scxentropy_loss(label_train,train_prob)
                # cat_total_loss += cat_loss

                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.cat_importance_weight_list[sidx])))

                total_loss = xentropy_loss + self.model_args["l1_lambda"]*l1_loss
        
            #Now we have total classification loss, lets update the gradient
            grads = tape.gradient(total_loss,self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.trainable_weights)
            )

            #Getting the validation accuracy for this category
            # valid_prob = self.get_sentiment_pred_prob_topic_basis(idx_valid,mask_valid,gate_tensor,sidx)
            valid_prob = self.get_sentiment_pred_prob_topic_direct(idx_valid,sidx)
            self.sent_valid_acc_list[sidx].update_state(label_valid,valid_prob)
    
            #Updating the metrics to track
            self.sent_pred_xentropy.update_state(xentropy_loss)
            self.sent_l1_loss_list[sidx].update_state(l1_loss)
        elif task=="topic":
            with tf.GradientTape() as tape:
                #Getting the bert representation
                bert_seq_output_train = self.get_bert_representation(
                                                    idx_train=idx_train,
                                                    mask_train=mask_train,
                )

                #Forward propagating the model
                train_prob,_ = self.get_topic_pred_prob(bert_seq_output_train,mask_train,gate_tensor,sidx)
                
                #Getting the loss for this classifier
                xentropy_loss = scxentropy_loss(label_train,train_prob)
                # cat_total_loss += cat_loss
                
                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.topic_importance_weight_list[sidx])))

                total_loss = xentropy_loss + self.model_args["l1_lambda"]*l1_loss
                
        
            #Now we have total classification loss, lets update the gradient
            grads = tape.gradient(total_loss,self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.trainable_weights)
            )

            #Getting the validation accuracy for this category
            bert_seq_output_valid = self.get_bert_representation(
                                                    idx_valid,
                                                    mask_valid,
            )
            valid_prob,_ = self.get_topic_pred_prob(bert_seq_output_valid,mask_valid,gate_tensor,sidx)
            self.topic_valid_acc_list[sidx].update_state(label_valid,valid_prob)
    
            #Updating the metrics to track
            self.topic_pred_xentropy.update_state(xentropy_loss)
            self.topic_l1_loss_list[sidx].update_state(l1_loss)
        else:
            raise NotImplementedError()



        # #Now we will train the topics for the dataset
        # all_topic_data = [(
        #                     cat_dataset_list[cat]["topic"],
        #                     cat_dataset_list[cat]["input_idx"],
        #                     cat_dataset_list[cat]["attn_mask"],
        #                     cat_dataset_list[cat]["topic_weight"]
        #             )
        #                 for cat in self.data_args["cat_list"]
        # ]
        # #Shuffling the topics for now (since one cat is ont topic for now)
        # topic,input_idx,attn_mask,topic_weight = zip(*all_topic_data)

        # topic_label_all     =   tf.concat(topic,axis=0)
        # input_idx           =   tf.concat(input_idx,axis=0)
        # attn_mask           =   tf.concat(attn_mask,axis=0)
        # topic_weight_all    =   tf.concat(topic_weight,axis=0)

        # #Shuffling the examples, incase the topic labels are skewed in a category
        # if(self.model_args["shuffle_topic_batch"]):
        #     #To shuffle, dont use random shuffle independently
        #     num_samples = tf.shape(topic_label_all).numpy()[0]
        #     perm_indices = np.expand_dims(np.random.permutation(num_samples),axis=-1)

        #     #Now we will gather the tensors using this perm indices
        #     topic_label_all = tf.gather_nd(topic_label_all,indices=perm_indices)
        #     input_idx       = tf.gather_nd(input_idx,indices=perm_indices)
        #     attn_mask       = tf.gather_nd(attn_mask,indices=perm_indices)
        

        # #Training the topic classifier in batches
        # topic_total_loss = 0.0
        # for iidx in range(len(self.data_args["cat_list"])):
        #     #Sharding the topic data into batches (to keep the batch size equal to topic ones)
        #     batch_size = self.data_args["batch_size"]
        #     topic_label = topic_label_all[iidx*batch_size:(iidx+1)*batch_size]
        #     topic_idx = input_idx[iidx*batch_size:(iidx+1)*batch_size]
        #     topic_mask = attn_mask[iidx*batch_size:(iidx+1)*batch_size]
        #     topic_weight = topic_weight_all[iidx*batch_size:(iidx+1)*batch_size]

        #     #Now we will iterate the training for each of the poic classifier
        #     for tidx,tname in enumerate(self.data_args["topic_list"]):
        #         #Taking aside a chunk of data for validation
        #         valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #         tclass_factor = self.data_args["per_topic_class"]
        #         #Getting the train data
        #         topic_label_train = topic_label[0:valid_idx,tidx]
        #         topic_idx_train = topic_idx[0:valid_idx]
        #         topic_mask_train = topic_mask[0:valid_idx]
        #         topic_weight_train = topic_weight[0:valid_idx,tidx]

        #         #Getting the validation data
        #         topic_label_valid = topic_label[valid_idx:,tidx]
        #         topic_idx_valid = topic_idx[valid_idx:]
        #         topic_mask_valid = topic_mask[valid_idx:]
        #         topic_weight_valid = topic_weight[valid_idx:,tidx]

        #         with tf.GradientTape() as tape:
        #             #Forward propagating the model
        #             topic_train_prob = self.get_topic_pred_prob(topic_idx_train,topic_mask_train,tidx)

        #             #Getting the loss for this classifier
        #             topic_loss = cxentropy_loss(topic_label_train,
        #                                         topic_train_prob,
        #                                         sample_weight=topic_weight_train,
        #             )
        #             topic_total_loss += topic_loss
            
        #         #Now we have total classification loss, lets update the gradient
        #         #TODO: BERT Model weight shouldn't be updated here. Correct this (used tf.stop_grad)
        #         grads = tape.gradient(topic_loss,self.trainable_weights)
        #         self.optimizer.apply_gradients(
        #             zip(grads,self.trainable_weights)
        #         )

        #         #Getting the validation accuracy for this category
        #         topic_valid_prob = self.get_topic_pred_prob(topic_idx_valid,topic_mask_valid,tidx)

        #         #For calculating the accuracy we need one single label instead of mixed prob.
        #         topic_label_valid_MAX = tf.argmax(topic_label_valid,axis=-1)
        #         self.topic_valid_acc_list[tidx].update_state(
        #                                             topic_label_valid_MAX,
        #                                             topic_valid_prob,
        #                                             sample_weight=topic_weight_valid
        #                                         )
        
        # #Updating the metrics to track
        # self.topic_pred_xentropy.update_state(topic_total_loss)


        # #Getting the tracking results
        # track_dict={}
        # track_dict["xentropy_sent"]=self.sent_pred_xentropy.result()
        # track_dict["xentorpy_topic"]=self.topic_pred_xentropy.result()

        # #Function to all the accuracy to tracker
        # def add_accuracy_mertrics(track_dict,acc_prefix,list_name,acc_list):
        #     for ttidx,ttname in enumerate(self.data_args[list_name]):
        #         track_dict["{}_{}".format(acc_prefix,ttidx)]=acc_list[ttidx].result()
        
        # #Adding the accuracies
        # add_accuracy_mertrics(track_dict=track_dict,
        #                         acc_prefix="sent_vacc",
        #                         list_name="cat_list",
        #                         acc_list=self.sent_valid_acc_list
        # )
        # add_accuracy_mertrics(track_dict=track_dict,
        #                         acc_prefix="topic_vacc",
        #                         list_name="topic_list",
        #                         acc_list=self.topic_valid_acc_list
        # )

        # return track_dict
        return None

    def valid_step_stage1(self,sidx,single_ds,gate_tensor,acc_op,task):
        '''
        Given a single dataset we will get the validation score
        (using the lower end of the dataset which was not used for training)
        '''
        label = single_ds["label"]
        idx = single_ds["topic_feature"]
        # mask = single_ds["attn_mask"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        # mask_valid = mask[valid_idx:]

        #Now we will make the forward pass
        if task=="sentiment":
            # valid_prob = self.get_sentiment_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
            valid_prob = self.get_sentiment_pred_prob_topic_direct(idx_valid,sidx)
        elif task=="topic":
            valid_prob = self.get_topic_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
        else:
            raise NotImplementedError()

        #Getting the validation accuracy
        acc = tf.keras.metrics.sparse_categorical_accuracy(label_valid,valid_prob)
        acc_op.update_state(acc)
    
    def train_step_stage2_advserial(self,cidx,tidx,single_ds,task,reverse_grad):
        '''
        This function will run one step of training for the classifier
        '''
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # cxentropy_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

        # #Get the dataset for each category
        # cat_dataset_list = data

        #Keeping track of classification accuracy
        cat_accuracy_list = []

        #Now getting the classification loss one by one each category        
        # cat_total_loss = 0.0
        # for cidx,cat in enumerate(self.data_args["cat_list"]):
        label = single_ds["label"]
        idx = single_ds["input_idx"]
        mask = single_ds["attn_mask"]
        topic_label = single_ds["topic_label"]#Only the debug topic label is coming here

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the train data
        label_train = label[0:valid_idx]
        idx_train = idx[0:valid_idx]
        mask_train = mask[0:valid_idx]
        topic_label_train = topic_label[0:valid_idx]

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        mask_valid = mask[valid_idx:]
        topic_label_valid = topic_label[valid_idx:]

        if task=="sentiment":
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape() as tape:
                #Forward propagating the model
                # train_prob = self.get_sentiment_pred_prob_topic_basis(idx_train,mask_train,gate_tensor,sidx)
                train_prob,_ = self.get_sentiment_pred_prob(idx_train,mask_train,None,cidx)

                #Getting the loss for this classifier
                xentropy_loss = scxentropy_loss(label_train,train_prob)
                # cat_total_loss += cat_loss

                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.cat_importance_weight_list[cidx])))

                total_loss = xentropy_loss + self.model_args["l1_lambda"]*l1_loss
        
            #Now we have total classification loss, lets update the gradient
            grads = tape.gradient(total_loss,self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.trainable_weights)
            )

            #Getting the validation accuracy for this category
            # valid_prob = self.get_sentiment_pred_prob_topic_basis(idx_valid,mask_valid,gate_tensor,sidx)
            valid_prob,_ = self.get_sentiment_pred_prob(idx_valid,mask_valid,None,cidx)
            self.sent_valid_acc_list[cidx].update_state(label_valid,valid_prob)
    
            #Updating the metrics to track
            self.sent_pred_xentropy.update_state(xentropy_loss)
            self.sent_l1_loss_list[cidx].update_state(l1_loss)
        elif task=="remove_topic":
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape(persistent=True) as tape:
                #Forward propagating the model
                # train_prob = self.get_sentiment_pred_prob_topic_basis(idx_train,mask_train,gate_tensor,sidx)
                train_prob,sentiment_emb = self.get_sentiment_pred_prob(idx_train,mask_train,None,cidx)
                
                #Getting the topic prediction
                topic_train_prob = self.get_topic_pred_prob_from_sentiment_emb(sentiment_emb,tidx,reverse_grad)

                #Getting the loss for this classifier
                xentropy_loss = scxentropy_loss(label_train,train_prob)
                # cat_total_loss += cat_loss

                #Getting the loss for this topic classfier
                topic_xentropy_loss = scxentropy_loss(topic_label_train,topic_train_prob)

                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.cat_importance_weight_list[cidx])))

                total_loss = xentropy_loss -topic_xentropy_loss + self.model_args["l1_lambda"]*l1_loss
            
            #Now we have total classification loss, lets update the gradient
            # grads = tape.gradient(total_loss,self.trainable_weights)
            # self.optimizer.apply_gradients(
            #     zip(grads,self.trainable_weights)
            # )

            #Getting the weight just from the our generator layers
            generator_trainable_weights = [ 
                            self.cat_importance_weight_list[cidx],       
            ] + self.cat_classifier_list[cidx].trainable_variables + self.cat_emb_layer_list[cidx].trainable_variables

            #Now we have total classification loss, lets update the gradient
            grads = tape.gradient(total_loss,generator_trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,generator_trainable_weights)
            )

            #Getting the weights from the discriminator
            discriminator_trainable_weights = [
                            self.topic_importance_weight_list[tidx],       
            ] + self.topic_classifier_list[tidx].trainable_variables
            grads = tape.gradient(topic_xentropy_loss,discriminator_trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,discriminator_trainable_weights)
            )

            #Getting the validation accuracy for this category
            # valid_prob = self.get_sentiment_pred_prob_topic_basis(idx_valid,mask_valid,gate_tensor,sidx)
            valid_prob,sentiment_emb = self.get_sentiment_pred_prob(idx_valid,mask_valid,None,cidx)
            self.sent_valid_acc_list[cidx].update_state(label_valid,valid_prob)

            topic_valid_prob = self.get_topic_pred_prob_from_sentiment_emb(sentiment_emb,tidx,reverse_grad)
            self.topic_valid_acc_list[tidx].update_state(topic_label_valid,topic_valid_prob)
    
            #Updating the metrics to track
            self.sent_pred_xentropy.update_state(xentropy_loss)
            self.topic_pred_xentropy.update_state(topic_xentropy_loss)
            self.sent_l1_loss_list[cidx].update_state(l1_loss)
        else:
            raise NotImplementedError()
    
    def valid_step_stage2_adversarial(self,sidx,single_ds,gate_tensor,acc_op,task):
        '''
        This validation step will be used by the stage 2 of our debugger
        '''
        label = single_ds["label"]
        idx = single_ds["input_idx"]
        mask = single_ds["attn_mask"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        mask_valid = mask[valid_idx:]

        #Now we will make the forward pass
        raise NotImplementedError()
        if task=="sentiment":
            # valid_prob = self.get_sentiment_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
            valid_prob = self.get_sentiment_pred_prob_topic_direct(idx_valid,sidx)
        elif task=="topic":
            valid_prob = self.get_topic_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
        else:
            raise NotImplementedError()

        #Getting the validation accuracy
        acc = tf.keras.metrics.sparse_categorical_accuracy(label_valid,valid_prob)
        acc_op.update_state(acc)
    
    def train_step_stage2_inlp(self,bidx,dataset_batch,task,P_matrix,cidx,tidx):
        '''
        This function will run one step of training for the classifier
        bidx: the batch idx of dataset which we will cache for now for
                faster training. This assumes that the btches are 
                not randomized and comes in different order every time.
                
        cidx: choose which category dataset to train the main classifier
        tidx: the topic index
        '''
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # cxentropy_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

        # #Get the dataset for each category
        # cat_dataset_list = data

        #Keeping track of classification accuracy
        cat_accuracy_list = []

        #Now getting the classification loss one by one each category        
        # cat_total_loss = 0.0
        # for cidx,cat in enumerate(self.data_args["cat_list"]):
        label = dataset_batch["label"]
        idx = dataset_batch["input_idx"]
        mask = dataset_batch["attn_mask"]
        topic_label = dataset_batch["topic_label"]#Only the debug topic label is coming here

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the train data
        label_train = label[0:valid_idx]
        idx_train = idx[0:valid_idx]
        mask_train = mask[0:valid_idx]
        if "topic" in task:
            topic_label_train = topic_label[0:valid_idx,tidx]

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        mask_valid = mask[valid_idx:]
        if "topic" in task:
            topic_label_valid = topic_label[valid_idx:,tidx]

        if "main" in task:
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape() as tape:
                #Forward propagating the model
                # train_prob = self.get_sentiment_pred_prob_topic_basis(idx_train,mask_train,gate_tensor,sidx)
                _,train_enc = self.get_sentiment_pred_prob(bidx,"train",idx_train,mask_train,None,cidx)
                #Passing this through the projection layer
                train_proj = self._get_proj_X_enc(train_enc,P_matrix)
                #Now getting the predction probability
                train_prob = self.get_main_pred_prob_final(train_proj,cidx)

                #Getting the loss for this classifier
                xentropy_loss = scxentropy_loss(label_train,train_prob)
                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.cat_importance_weight_list[cidx])))
                total_loss = xentropy_loss #+ self.model_args["l1_lambda"]*l1_loss

            if task=="main":
                #Now we have total classification loss, lets update the gradient
                grads = tape.gradient(total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )
            elif task=="main_inlp":
                #Just update the classifier weights
                main_head_params = [] + self.cat_classifier_list[cidx].trainable_variables
                grads = tape.gradient(total_loss,main_head_params)
                self.optimizer.apply_gradients(
                    zip(grads,main_head_params)
                )
            else:
                raise NotImplementedError()

            #Getting the validation accuracy for this category
            # valid_prob = self.get_sentiment_pred_prob_topic_basis(idx_valid,mask_valid,gate_tensor,sidx)
            _,valid_enc = self.get_sentiment_pred_prob(bidx,"valid",idx_valid,mask_valid,None,cidx)
            valid_proj = self._get_proj_X_enc(valid_enc,P_matrix)
            valid_prob = self.get_main_pred_prob_final(valid_proj,cidx)
            self.sent_valid_acc_list[cidx].update_state(label_valid,valid_prob)
    
            #Updating the metrics to track
            self.sent_pred_xentropy_list[cidx].update_state(xentropy_loss)
            self.sent_l1_loss_list[cidx].update_state(l1_loss)
        elif "topic" in task:
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape() as tape:
                #Forward propagating the model
               #We need to use the same encoder pipeline as the main task hence the sentiment pred_prob func
                _,train_enc = self.get_sentiment_pred_prob(bidx,"train",idx_train,mask_train,None,cidx)
                train_proj = self._get_proj_X_enc(train_enc,P_matrix)
                topic_train_prob = self.get_topic_pred_prob_final(train_proj,tidx)
                
                xentropy_loss = scxentropy_loss(topic_label_train,topic_train_prob)
                l1_loss = tf.reduce_sum(tf.abs(tf.sigmoid(self.cat_importance_weight_list[cidx])))
                topic_total_loss = xentropy_loss #+ self.model_args["l1_lambda"]*l1_loss
            
            #Now training the encoder to also retain the topic related info if want to have them
            if task=="topic":
                grads = tape.gradient(topic_total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )
            elif task=="topic_inlp":
                #Here we will only train the topic classiifer
                topic_params = [] + self.topic_classifier_list[tidx].trainable_variables
                grads = tape.gradient(topic_total_loss,topic_params)
                self.optimizer.apply_gradients(
                    zip(grads,topic_params)
                )
            else:
                raise NotImplementedError()

            #Getting the validation accuracy for this category
            _,valid_enc = self.get_sentiment_pred_prob(bidx,"valid",idx_valid,mask_valid,None,cidx)
            valid_proj = self._get_proj_X_enc(valid_enc,P_matrix)
            topic_valid_prob = self.get_topic_pred_prob_final(valid_proj,tidx)

            
            #Updating the metrics to track
            self.topic_valid_acc_list[tidx].update_state(topic_label_valid,topic_valid_prob)
            self.topic_pred_xentropy_list[tidx].update_state(xentropy_loss)
            self.sent_l1_loss_list[cidx].update_state(l1_loss)
        else:
            raise NotImplementedError()
    
    def valid_step_stage2_inlp(self,bidx,dataset_batch,P_matrix,cidx,tidx):
        '''
        This validation step will be used by the stage 2 of our debugger
        '''
        label = dataset_batch["label"]
        idx = dataset_batch["input_idx"]
        mask = dataset_batch["attn_mask"]
        topic_label = dataset_batch["topic_label"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        mask_valid = mask[valid_idx:]
        topic_label_valid = topic_label[valid_idx:,tidx]

        #Passing the input through the encoder
        _,valid_enc = self.get_sentiment_pred_prob(bidx,"valid",idx_valid,mask_valid,None,cidx)
        valid_proj = self._get_proj_X_enc(valid_enc,P_matrix)

        #Now we will make the forward pass and get the validation accuracy
        valid_prob = self.get_main_pred_prob_final(valid_proj,cidx)
        self.sent_valid_acc_list[cidx].update_state(label_valid,valid_prob)
    
        topic_valid_prob = self.get_topic_pred_prob_final(valid_proj,tidx)
        self.topic_valid_acc_list[tidx].update_state(topic_label_valid,topic_valid_prob)
    
    def _get_proj_X_enc(self,X_enc,P_matrix):
        '''
        This will give the projected encoded latent representaion which will be furthur
        removed of the information.
        '''
        #Converting to the numpy array
        P_matrix = tf.constant(P_matrix.T,dtype=tf.float32)

        #Now projecting this latent represention into null space
        X_proj = tf.matmul(X_enc,P_matrix)

        return X_proj
    
    def get_angle_between_classifiers(self,class_idx):
        '''
        A utility function that will tell us where the convergence of each of the classifier
        is happening with respect to one another since in high definition it dosent
        make sense to choose any one reference point also.

        Problems:
        1. Here we could have multi-label classifier so we may not have just one direction
            to remove + which direction to take into consideration?
            (Try calculating the angle within the pos+neg class)

        '''
        #Getting the classifier angles
        classifier_dict = {
            "topic{}".format(tidx) : self.topic_classifier_list[tidx].get_weights()[0][:,class_idx]
                for tidx in range(self.data_args["num_topics"])
        }
        classifier_dict["main"] = self.cat_classifier_list[self.data_args["debug_cidx"]].get_weights()[0][:,class_idx]

        #Getting the norm of the classifier
        classifier_norm = {
            cname:np.linalg.norm(classifier_dict[cname])
                for cname in classifier_dict.keys()
        }
        print("Classifiers Norm:")
        mypp(classifier_norm)

        #Now its time to get the angle among each of them
        convergence_angle=defaultdict(dict)
        for cname1,w1 in classifier_dict.items():
            for cname2,w2 in classifier_dict.items():
                norm1 = np.linalg.norm(w1)
                norm2 = np.linalg.norm(w2)

                angle = np.arccos(np.sum(w1*w2)/(norm1*norm2))/np.pi
                convergence_angle[cname1][cname2]=float(angle) 
        print("Convergence Angle: (in multiple of pi)")
        mypp(convergence_angle)

        return convergence_angle
    
    def get_all_classifier_accuracy(self,):
        '''
        '''
        classifier_accuracy = {}
        classifier_accuracy["main"]=float(self.sent_valid_acc_list[self.data_args["debug_cidx"]].result().numpy())
        for tidx in range(self.data_args["num_topics"]):
            classifier_accuracy["topic{}".format(tidx)]=float(self.topic_valid_acc_list[tidx].result().numpy())
        
        return classifier_accuracy
    
    

class SimpleNBOW(keras.Model):
    '''
    This will be used for the Stage 2 when we want to start with simple NBOW model
    '''
    def __init__(self,data_args,model_args,data_handler):
        super(SimpleNBOW,self).__init__()
        self.data_args = data_args 
        self.model_args = model_args
        self.data_handler = data_handler
        self.init_weights = defaultdict()
        self.small_epsilon = 1e-10

        #Saving the experiment configuration to a file
        self.save_the_experiment_config()

        #Initilaize the proble metric list
        self.probe_metric_list=[]

        #Now initilaizing some of the layers for encoder
        if self.data_args["dtype"]=="toynlp2":
            self.pre_encoder_layer = self.get_nbow_avg_layer()
        elif self.data_args["dtype"]=="toytabular2":
            self.pre_encoder_layer = self.get_dummy_linear_layer()
        elif self.model_args["bert_as_encoder"]==True:
            #Getting the "pre-encoder" layer
            self.pre_encoder_layer = self.get_bert_encoder_layer()
        else:
            raise NotImplementedError()
        
        # #Setting the gpu
        # self.set_gpu()
        


        #Initialiing the hidden layer after encoding
        self.hidden_layer_list = []
        self.dropout_layer_list = []
        if self.model_args["num_hidden_layer"]!=0:
            if self.data_args["dtype"]=="toynlp2":
                self.hlayer_dim = 50
            elif self.data_args["dtype"]=="toytabular2":
                self.hlayer_dim = self.data_args["inv_dims"]+self.data_args["sp_dims"]
        else:
            if self.model_args["bert_as_encoder"]==True:
                self.hlayer_dim = self.model_args["bemb_dim"]
            elif self.data_args["dtype"]=="toynlp2":
                self.hlayer_dim = self.emb_dim
            elif self.data_args["dtype"]=="toytabular2":
                self.hlayer_dim = self.data_args["inv_dims"]+self.data_args["sp_dims"]
        
        for _ in range(self.model_args["num_hidden_layer"]):
            #Getting the dense hidden layer
            hlayer = layers.Dense(
                            self.hlayer_dim,
                            activation="relu",
                            kernel_regularizer=tf.keras.regularizers.l2(self.model_args["l2_lambd"]),
            )
            self.hidden_layer_list.append(hlayer)

            #Getting the dropout layer (to be applied after hidden layer)
            dlayer = layers.Dropout(
                            self.model_args["dropout_rate"],
                            input_shape=(self.hlayer_dim,),
            )
            self.dropout_layer_list.append(dlayer)
        
        self.last_layer_activation=None
        self.last_layer_dim = None
        if self.model_args["loss_type"]=="linear_svm":
            self.last_layer_activation=None
            self.last_layer_dim=1
        elif self.model_args["loss_type"]=="x_entropy":
            self.last_layer_activation="softmax"
            self.last_layer_dim=2
        else:
            raise NotImplementedError()

        #Initializing the classifier for the main task
        self.main_task_classifier = layers.Dense(
                            self.last_layer_dim,
                            activation=self.last_layer_activation,
                            kernel_regularizer=tf.keras.regularizers.l2(self.model_args["l2_lambd"]),
        )
        #Initializing some of the metrics for main task
        self.main_pred_xentropy = keras.metrics.Mean(name="sent_pred_x")
        self.main_valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="main_vacc")

        ###############################################################
        #CAUTION : ADD a reset metric Function for every metric added
        ###############################################################
        #Creating the topic classifiers
        self.topic_task_classifier_list = []
        self.topic_pred_xentropy_list = []
        self.topic_valid_accuracy_list = []
        self.topic_flip_main_valid_accuracy_list = [] #Main Accuracy we get on flipping the feature/topic
        self.topic_flip_main_prob_delta_list = [] #Keep the mean probabilty change happened to the examples
        #since the delta_p will have smallar changes if the activation are very high which is made by the x-entorpy obj.
        self.topic_flip_main_logprob_delta_list = [] #Keeps the difference between the logprob of the output
        self.topic_flip_emb_diff_list = []   #Keep track of absolute change in the embedding 
        self.topic_smin_main_valid_accuracy_list = []
        for tidx in range(self.data_args["num_topics"]):
            #Initializing the classifier for the topic task
            self.topic_task_classifier_list.append(
                layers.Dense(
                            self.last_layer_dim,
                            activation=self.last_layer_activation,
                            kernel_regularizer=tf.keras.regularizers.l2(self.model_args["l2_lambd"]),
                )
            )

            #Initializing the metrics for the topic task
            self.topic_pred_xentropy_list.append(keras.metrics.Mean(name="topic_{}_spred_x".format(tidx)))
            #TODO: Is this updating by taking mean or overwriting with completely new value? Looks like mean wrapper in source-code
            self.topic_valid_accuracy_list.append(tf.keras.metrics.SparseCategoricalAccuracy(name="topic_{}_vacc".format(tidx)))
            
            #To keep the main accuracy when we flip the features in the input
            self.topic_flip_main_valid_accuracy_list.append(tf.keras.metrics.SparseCategoricalAccuracy(name="topic_{}_flip_main_vacc".format(tidx)))
            self.topic_smin_main_valid_accuracy_list.append(tf.keras.metrics.SparseCategoricalAccuracy(name="topic_{}_smin_main_vacc".format(tidx)))
            self.topic_flip_main_prob_delta_list.append(keras.metrics.Mean(name="topic_{}_flip_main_prob_delta".format(tidx)))
            self.topic_flip_main_logprob_delta_list.append(keras.metrics.Mean(name="topic_{}_flip_main_logprob_delta".format(tidx)))
            self.topic_flip_emb_diff_list.append(tf.keras.metrics.Mean(name="topic_{}_flip_emb_delta".format(tidx)))

    def get_all_head_init_weights(self,):
        '''
        This will get the initialized weights which can be used later for
        reinitialization of the classifier.

        Make sure the weights are initialized i.e atleast one iter is made
        '''
        #Dont update the init weights if we already have them
        if(len(self.init_weights)!=0):
            return
        
        #Getting the main classifier weights
        self.init_weights["main"]=self.main_task_classifier.get_weights()
        #Getting the topic classifier weights
        for tidx in range(self.data_args["num_topics"]):
            self.init_weights["topic{}".format(tidx)]=self.topic_task_classifier_list[tidx].get_weights()
        
    def set_all_head_init_weights(self,):
        '''
        Here we will reinitialize all the weights in the task classifier with the one
        we had in the begining of the training.
        '''
        #Reinitializing the main task classifier
        self.main_task_classifier.set_weights(self.init_weights["main"])
        #Reinitializing the topic classifier weights
        for tidx in range(self.data_args["num_topics"]):
            self.topic_task_classifier_list[tidx].set_weights(
                        self.init_weights["topic{}".format(tidx)]
            )

    def compile(self, optimizer):
        super(SimpleNBOW, self).compile()
        self.optimizer = optimizer 

    def reset_all_metrics(self,):
        #Resetting the main task related metrics
        self.main_pred_xentropy.reset_states()
        self.main_valid_accuracy.reset_states()

        #Ressting the topic related metrics
        for tidx in range(self.data_args["num_topics"]):
            self.topic_pred_xentropy_list[tidx].reset_states()
            self.topic_valid_accuracy_list[tidx].reset_states()
            self.topic_flip_main_valid_accuracy_list[tidx].reset_states()
            self.topic_smin_main_valid_accuracy_list[tidx].reset_states()
            self.topic_flip_main_prob_delta_list[tidx].reset_states()
            self.topic_flip_main_logprob_delta_list[tidx].reset_states()
            self.topic_flip_emb_diff_list[tidx].reset_states()

    def get_nbow_avg_layer(self,):
        '''
        This will create a new layer which will be used for averaging the 
        bag of words correctly i.e by ignoring the unk value
        '''
        class NBOWAvgLayer(keras.layers.Layer):
            '''
            '''
            def __init__(self,embedding_layer,embedding_weight_layer,unk_widx,normalize_emb):
                super(NBOWAvgLayer,self).__init__()
                self.embeddingLayerMain = embedding_layer
                self.embeddingLayerWeight = embedding_weight_layer
                self.unk_widx = unk_widx
                self.normalize_emb = normalize_emb

            def call(self,X_input,input_mask):
                '''
                X_inputs: This is the word --> idx 
                X_emb   : This is the idx --> emb vectors
                input_mask : not used. Put here for unoformity across pre-encoder
                '''
                #Creating our own mask and get number of non zero
                non_unk_mask=tf.cast(
                        tf.not_equal(X_input,self.unk_widx),
                    dtype=tf.float32,
                )
                num_words=tf.reduce_sum(non_unk_mask,axis=1,keepdims=True)

                #Now we will get the embedding of the inputs
                X_emb = self.embeddingLayerMain(X_input)
                #Normalizing the inputs
                if self.normalize_emb:
                    X_emb_norm = tf.math.l2_normalize(X_emb,axis=-1)
                else:
                    X_emb_norm = X_emb

                #Getting the weights of the individual words
                X_weight = self.embeddingLayerWeight(X_input)

                #Getting the nonunk_mask ready
                non_unk_mask = tf.expand_dims(non_unk_mask,axis=-1)
                #Adding all the nonunk vectors
                X_emb_weighted = X_emb_norm * tf.sigmoid(X_weight) * non_unk_mask
                #Now we need to take average of the embedding (zero vec non-train)
                X_bow=tf.divide(tf.reduce_sum(X_emb_weighted,axis=1,name="word_sum"),num_words)

                return X_bow
        
        #Get the embedding and weight matrix
        embedding_layer,embedding_weight_layer = self._initialize_embedding_layer()
        nbow_avg_layer = NBOWAvgLayer(
                                embedding_layer=embedding_layer,
                                embedding_weight_layer=embedding_weight_layer,
                                unk_widx=self.data_handler.emb_model.key_to_index["unk"],
                                normalize_emb=self.model_args["normalize_emb"]
        )

        return nbow_avg_layer
    
    def _initialize_embedding_layer(self,):
        '''
        '''
        #Normalizing the embedding matrix if we want to
        if(self.model_args["normalize_emb"]==True):
            self.data_handler.emb_model.unit_normalize_all()

        #Creating the embedding layer
        emb_matrix = self.data_handler.emb_model.vectors
        vocab_len , emb_dim = emb_matrix.shape
        self.emb_dim = emb_dim
        embedding_layer = layers.Embedding(
                                    vocab_len,
                                    emb_dim,
                                    input_length = self.data_args["max_len"],
                                    trainable = self.model_args["train_emb"]
        )
        #Building the embedding layer
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])



        #Creating the embedding weight layer
        emb_weight_matrix = np.random.randn(vocab_len,1)
        embedding_weight_layer = layers.Embedding(
                                    vocab_len,
                                    1,
                                    input_length = self.data_args["max_len"],
                                    trainable = True
        )
        embedding_weight_layer.build((None,))
        embedding_weight_layer.set_weights([emb_weight_matrix])

        return embedding_layer,embedding_weight_layer
    
    def get_dummy_linear_layer(self,):
        '''
        '''
        class DummyLinearLayer(tf.keras.layers.Layer):

            def __init__(self,):
                super(DummyLinearLayer,self).__init__()
            
            def build(self,input_shape):
                return 
            
            def call(self,inputs,input_mask):
                return inputs
        
        dummy_linear_layer = DummyLinearLayer()
        return dummy_linear_layer

    def get_bert_encoder_layer(self,):
        '''
        This will initiale the BERT model as a layer which couild be used to enode the data
        '''
        class BERTLayer(tf.keras.layers.Layer):
            def __init__(self,model_args):
                super(BERTLayer,self).__init__()
                #Initializing the BERT model as required
                if "distil" in data_args["transformer_name"]:
                        self.bert_model = TFDistilBertModel.from_pretrained(data_args["transformer_name"])
                else:
                    self.bert_model = TFBertModel.from_pretrained(data_args["transformer_name"])
                
                #Do we want to train the full BERT or not
                if model_args["train_bert"]==False:
                    for layer in self.bert_model.layers:
                        layer.trainable = False
            
            def build(self,input_shape):
                return
            
            def call(self,input_idx,attn_mask):
                #Getting the bert activation
                bert_outputs=self.bert_model(
                                        input_ids=input_idx,
                                        attention_mask=attn_mask
                )
                cls_output = bert_outputs.pooler_output
                return cls_output

        #Creating the BERT Layer and returning it
        bert_layer = BERTLayer(self.model_args)
        return bert_layer

    def _encoder(self,X_input,attn_mask=None,training=None):
        '''
        This function will be responsible to encode the input to the 
        latent space for all the paths.
        '''
        #Get average embedding
        Xproj = self.pre_encoder_layer(X_input,attn_mask)
        #Passing through the hidden layer
        for hlayer,dlayer in zip(self.hidden_layer_list,self.dropout_layer_list):
            #Passing through the hidden layer
            Xproj = hlayer(Xproj)
            #Passing through the dropout layer
            Xproj = dlayer(Xproj,training=training)
        
        return Xproj

    def get_main_task_pred_prob(self,X_enc):
        '''
        Forward propagate the input for the main tasssk
        '''
        # Xproj = self._encoder(X_input)
        #Getting the prediction
        Xpred = self.main_task_classifier(X_enc)

        return Xpred

    def get_max_margin_loss(self,Xmargin,label):
        '''
        This function will calculate the hinge loss from the final embeddeding 
        '''
        hinge_loss_op = tf.keras.losses.Hinge()
        label = tf.cast(label,tf.float32)
        hinge_loss = hinge_loss_op(label,Xmargin)
        return hinge_loss
    
    def get_max_margin_prob_vector(self,Xmargin):
        '''
        Here we will:
        1. Conver the margin in prob. naturally by sigmoid
        '''
        label1_prob = tf.math.sigmoid(Xmargin)
        label0_prob = 1-label1_prob

        prob_vec = tf.concat([label0_prob,label1_prob],axis=-1)
        return prob_vec

    def get_topic_pred_prob(self,X_enc,tidx):
        '''
        Forward propagate the input for the main tasssk
        '''
        # Xproj = self._encoder(X_input)
        #Getting the prediction
        Xpred = self.topic_task_classifier_list[tidx](X_enc)

        return Xpred
    
    def train_step_stage2(self,dataset_batch,task,P_matrix,cidx,adv_rm_tidx=None):
        '''
        cidx        : the classifier index we want to evaluate on (mainly for topic)
        adv_rm_tidx : the topic index which we want to remove simultaneously training the main head
                        (or should we do it in alreanating faishon?)

        '''
        #Getting the dataset splits for train and valid       
        label = dataset_batch["label"]
        idx = dataset_batch["input_idx"]
        topic_label = dataset_batch["topic_label"]#Only the debug topic is coming here (one rnow)

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the train data
        label_train = label[0:valid_idx]
        idx_train = idx[0:valid_idx]
        topic_label_train = topic_label[0:valid_idx,cidx]
        if adv_rm_tidx!=None:
            rm_topic_label_train = topic_label[0:valid_idx,adv_rm_tidx]
        
        #Getting the attention mask if we are training the BERT
        attn_mask_train=None
        if self.model_args["bert_as_encoder"]==True:
            attn_mask_train = dataset_batch["attn_mask"][0:valid_idx]


        #Initializing the loss metric
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        if task=="adv_rm_with_main":
            #Here we will train the main head along with the removing the topic information
            #from the latent space by the adversarial method
            with tf.GradientTape() as tape:
                #Encoding the input
                enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                #Getting the projection (I for main full traning)
                enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)

                #Get the prediction probability for the main task
                main_train_prob = self.get_main_task_pred_prob(enc_proj_train)
                #Getting the x-entropy lossss
                main_xentropy_loss = scxentropy_loss(label_train,main_train_prob)
                total_loss = main_xentropy_loss

                #Adding the gradient-reversal layer on enc_proj_train
                enc_proj_train_rev = grad_reverse(enc_proj_train)
                #Getting the adversarial loss from the latent space
                rm_topic_train_prob = self.get_topic_pred_prob(enc_proj_train_rev,adv_rm_tidx)
                #Getting the cross entropy loss for this topic classifier
                rm_topic_loss = scxentropy_loss(rm_topic_label_train,rm_topic_train_prob)

                #Getting the regularization loss
                regularization_loss = tf.math.add_n(self.losses)
                total_loss += regularization_loss


                #Adding to the total loss
                total_loss += self.model_args["rev_grad_strength"]*rm_topic_loss
            
            #Updating the total loss for main 
            self.main_pred_xentropy.update_state(main_xentropy_loss)
            #Updating the x-entropy loss for topic
            self.topic_pred_xentropy_list[adv_rm_tidx].update_state(rm_topic_loss)

            #Getting the gradient and updating the parameter
            grads = tape.gradient(total_loss,self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.trainable_weights)
            )

            #Getting the validation accuracy for the removal topic
            # enc_valid = self._encoder(idx_valid,training=False)
            # enc_proj_valid = self._get_proj_X_enc(enc_valid,P_matrix)
            # #Getting the accuracy for the main head
            # main_valid_prob = self.get_main_task_pred_prob(enc_proj_valid)
            # self.main_valid_accuracy.update_state(label_valid,main_valid_prob)
            # #Getting the accuracy for the topic which we want to remove
            # rm_topic_valid_prob = self.get_topic_pred_prob(enc_proj_valid,adv_rm_tidx)
            # self.topic_valid_accuracy_list[adv_rm_tidx].update_state(rm_topic_label_valid,rm_topic_valid_prob)
        elif "main" in task:
            #Training the main task
            if self.model_args["loss_type"]=="x_entropy":
                with tf.GradientTape() as tape:
                    #Encoding the input
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    #Getting the projection (I for main full traning)
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Get the prediction probability
                    main_train_prob = self.get_main_task_pred_prob(enc_proj_train)

                    #Getting the x-entropy lossss
                    main_xentropy_loss = scxentropy_loss(label_train,main_train_prob)

                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)

                    total_loss = main_xentropy_loss + regularization_loss

                    #Updating the total loss
                    self.main_pred_xentropy.update_state(main_xentropy_loss)
            elif self.model_args["loss_type"]=="linear_svm":
                '''
                Here we will train the model using the hinge loss used in the svm setting.
                '''
                with tf.GradientTape() as tape:
                    #Encoding the input will be same
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Getting non-softmaxed final projection
                    main_train_margin = self.get_main_task_pred_prob(enc_proj_train)

                    #Getting the svm_loss for this projection
                    main_svm_loss = self.get_max_margin_loss(Xmargin=main_train_margin,label=label_train)
                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)

                    total_loss = main_svm_loss + regularization_loss
                    #Updating the total loss
                    self.main_pred_xentropy.update_state(main_svm_loss)

            else:
                raise NotImplementedError()

            #Backpropagating
            if task=="main":
                grads = tape.gradient(total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )
            elif task=="inlp_main":
                #Only train the top head classifier
                main_head_params = [] + self.main_task_classifier.trainable_variables
                grads = tape.gradient(total_loss,main_head_params)
                self.optimizer.apply_gradients(
                    zip(grads,main_head_params)
                )
            else:
                raise NotImplementedError()

            #We wont measure validation accuracy here now on
            #Getting the validation loss/accuracy
            # enc_valid = self._encoder(idx_valid)
            # enc_proj_valid = self._get_proj_X_enc(enc_valid,P_matrix)
            # #Getting the accuracy for the main head
            # main_valid_prob = self.get_main_task_pred_prob(enc_proj_valid)
            # self.main_valid_accuracy.update_state(label_valid,main_valid_prob)
        elif task=="adv_rm_only_topic":
            #This could be used if we want to remove the topic adversarilly in an alternate
            #training faishon with the main task objective. So we dont need extra adv_rm_cidx
            if self.model_args["loss_type"]=="x_entropy":
                with tf.GradientTape() as tape:
                    #Encoding the input
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    #Getting the projection first
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Adding the gradient-reversal layer on enc_proj_train
                    enc_proj_train_rev = grad_reverse(enc_proj_train)
                    #Getting prediction probability
                    rm_topic_train_prob = self.get_topic_pred_prob(enc_proj_train_rev,cidx)
                    #Getting the x-entropy losssss for the topic
                    topic_xentropy_loss = scxentropy_loss(topic_label_train,rm_topic_train_prob)
                    topic_total_loss = topic_xentropy_loss

                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)
                    topic_total_loss += regularization_loss

                #Here will will train encoder also (to stop pulling out info in latent space)
                grads = tape.gradient(topic_total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )

                #Updating the loss for the topic
                self.topic_pred_xentropy_list[cidx].update_state(topic_xentropy_loss)
            elif self.model_args["loss_type"]=="linear_svm":
                with tf.GradientTape() as tape:
                    #Encoding the input
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    #Getting the projection first
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Adding the gradient-reversal layer on enc_proj_train
                    enc_proj_train_rev = grad_reverse(enc_proj_train)
                    #Getting prediction probability
                    rm_topic_train_margin = self.get_topic_pred_prob(enc_proj_train_rev,cidx)
                    #Getting the svm loss
                    topic_svm_loss = self.get_max_margin_loss(
                                                Xmargin=rm_topic_train_margin,
                                                label=topic_label_train
                    )
                    topic_total_loss = topic_svm_loss

                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)
                    topic_total_loss += regularization_loss

                #Here will will train encoder also (to stop pulling out info in latent space)
                grads = tape.gradient(topic_total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )

                #Updating the loss for the topic
                self.topic_pred_xentropy_list[cidx].update_state(topic_svm_loss)

            #Now updating the validation accuracy
            # enc_valid = self._encoder(idx_valid)
            # enc_proj_valid = self._get_proj_X_enc(enc_valid,P_matrix)
            # topic_valid_prob = self.get_topic_pred_prob(enc_proj_valid,cidx)
            # self.topic_valid_accuracy_list[cidx].update_state(topic_label_valid,topic_valid_prob)

        elif "topic" in task:
            if self.model_args["loss_type"]=="x_entropy":
                with tf.GradientTape() as tape:
                    #Encoding the input
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    #Getting the projection first
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Here we will train the topic classifier
                    topic_train_prob = self.get_topic_pred_prob(enc_proj_train,cidx)

                    #Getting the x-entropy losssss for the topic
                    topic_xentropy_loss = scxentropy_loss(topic_label_train,topic_train_prob)
                    topic_total_loss = topic_xentropy_loss

                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)
                    topic_total_loss += regularization_loss
                    #Updating the xentropy loss for the topic
                    self.topic_pred_xentropy_list[cidx].update_state(topic_xentropy_loss)
            elif self.model_args["loss_type"]=="linear_svm":
                '''
                Here we will train the model using the hinge loss used in the svm setting.
                '''
                with tf.GradientTape() as tape:
                    #Encoding the input will be same
                    enc_train = self._encoder(idx_train,attn_mask=attn_mask_train,training=True)
                    enc_proj_train = self._get_proj_X_enc(enc_train,P_matrix)
                    #Getting non-softmaxed final projection
                    topic_train_margin = self.get_topic_pred_prob(enc_proj_train,cidx)

                    #Getting the svm_loss for this projection
                    topic_svm_loss = self.get_max_margin_loss(
                                                Xmargin=topic_train_margin,
                                                label=topic_label_train
                    )
                    #Getting the regularization loss
                    regularization_loss = tf.math.add_n(self.losses)

                    topic_total_loss = topic_svm_loss + regularization_loss
                    #Updating the xentropy loss for the topic
                    self.topic_pred_xentropy_list[cidx].update_state(topic_svm_loss)
            else:
                raise NotImplementedError()
            
            #Get the topic classifier parameters
            if(task=="inlp_topic"):
                #Here we will just train the topic classifier only
                topic_params = [] + self.topic_task_classifier_list[cidx].trainable_variables
                grads = tape.gradient(topic_total_loss,topic_params)
                self.optimizer.apply_gradients(
                    zip(grads,topic_params)
                )
            elif(task=="topic"):
                #Here will will train encoder also (in pull out info in latent space)
                grads = tape.gradient(topic_total_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )
            else:
                raise NotImplementedError()

            #We wont measure the validation loss here from now on
            #Getting the validation loss for the topic
            # enc_valid = self._encoder(idx_valid)
            # enc_proj_valid = self._get_proj_X_enc(enc_valid,P_matrix)
            # topic_valid_prob = self.get_topic_pred_prob(enc_proj_valid,cidx)
            # self.topic_valid_accuracy_list[cidx].update_state(topic_label_valid,topic_valid_prob)
        else:
            raise NotImplementedError()

    def valid_step_stage2(self,dataset_batch,P_matrix,cidx):
        '''
        '''
        #Resetting all the metrics
        # self.reset_all_metrics() this leads to empty metrci at end. Why?

        #Getting the dataset splits for train and valid       
        label = dataset_batch["label"]
        idx = dataset_batch["input_idx"]
        topic_label = dataset_batch["topic_label"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )
        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        topic_label_valid = topic_label[valid_idx:,cidx]
        #Skipping if we dont have enough samples
        if(idx_valid.shape[0]==0):
            return
        
        #Getting the attention mask if we are validating the BERT
        attn_mask_valid=None
        if self.model_args["bert_as_encoder"]==True:
            attn_mask_valid = dataset_batch["attn_mask"][valid_idx:]


        #Getting the latent representaiton for the input
        X_latent = self._encoder(idx_valid,attn_mask=attn_mask_valid,training=False)
        X_proj = self._get_proj_X_enc(X_latent,P_matrix)



        #Getting the Smin group mask for this topic
        smin_mask = tf.math.logical_or(
                                    tf.math.logical_and(
                                                            label_valid==1,
                                                            topic_label_valid==0,
                                    ),
                                    tf.math.logical_and(
                                                            label_valid==0,
                                                            topic_label_valid==1,
                                    ),
        )

        if self.model_args["loss_type"]=="x_entropy":
            #Getting the validation accuracy of the main task
            #TODO: This assumes that this is the last layer of the both branches
            main_valid_prob = self.main_task_classifier(X_proj)
            self.main_valid_accuracy.update_state(label_valid,main_valid_prob)
            #Getting the accuracy in Smin group
            self.topic_smin_main_valid_accuracy_list[cidx].update_state(
                                                        label_valid[smin_mask],
                                                        main_valid_prob[smin_mask]
            )

            #Getting the topic validation accuracy
            topic_valid_prob = self.topic_task_classifier_list[cidx](X_proj)
            self.topic_valid_accuracy_list[cidx].update_state(topic_label_valid,topic_valid_prob)
        elif self.model_args["loss_type"]=="linear_svm":
            #Getting the main task accuracy
            main_valid_margin = self.main_task_classifier(X_proj)
            #Getting the probability from the margin
            main_valid_prob = self.get_max_margin_prob_vector(main_valid_margin)
            self.main_valid_accuracy.update_state(label_valid,main_valid_prob)
            #Getting the accuracy in Smin group
            self.topic_smin_main_valid_accuracy_list[cidx].update_state(
                                                        label_valid[smin_mask],
                                                        main_valid_prob[smin_mask]
            )

            #Getting the topic accuracy
            topic_valid_margin = self.topic_task_classifier_list[cidx](X_proj)
            topic_valid_prob = self.get_max_margin_prob_vector(topic_valid_margin)
            self.topic_valid_accuracy_list[cidx].update_state(topic_label_valid,topic_valid_prob)
        else:
            raise NotImplementedError()

        # mmd_var_dict=dict(
        #                 main_valid_prob = main_valid_prob,
        #                 main_label = label_valid,
        #                 topic_label = topic_label_valid
        # )
        return None
    
    def valid_step_stage2_flip_topic(self,dataset_batch,P_matrix,cidx):
        '''
        This function will calculate the main classifier accuracy given we flip the topic cidx
        in the input space.

        This will be proxy for feature weight coefficient of the main classifier since the
        direction of feature is not known.
        '''
        #Getting the dataset splits for train and valid       
        label = dataset_batch["label"]
        idx = dataset_batch["input_idx"]
        flip_idx = dataset_batch["input_idx_t{}_flip".format(cidx)]
        topic_label = dataset_batch["topic_label"]

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )
        #Getting the validation data
        idx_valid = idx[valid_idx:]
        label_valid = label[valid_idx:]
        flip_idx_valid = flip_idx[valid_idx:]
        
        #Skipping if we dont have enough samples
        if(idx_valid.shape[0]==0):
            return

        #Getting the attention mask if we are validating the BERT
        attn_mask_valid=None
        flip_attn_mask_valid=None
        if self.model_args["bert_as_encoder"]==True:
            attn_mask_valid = dataset_batch["attn_mask"][valid_idx:]
            flip_attn_mask_valid = dataset_batch["attn_mask_t{}_flip".format(cidx)][valid_idx:]


        #Getting the probability of main label form the actual input
        #Getting the latent representaiton for the flipped input
        X_latent = self._encoder(idx_valid,attn_mask=attn_mask_valid,training=False)
        X_proj_actual = self._get_proj_X_enc(X_latent,P_matrix)

        #Getting the latent representaiton for the flipped input
        X_latent = self._encoder(flip_idx_valid,attn_mask=flip_attn_mask_valid,training=False)
        X_proj_flip = self._get_proj_X_enc(X_latent,P_matrix)


        def get_measure_flip_mask():
            '''
            If we dont want to measure the flip delta_prob in some type of example
            '''
            #Getting the lebel for which we are flipping
            topic_label_valid = topic_label[valid_idx:,cidx]

            if self.data_args["neg1_flip_method"]=="dont_measure":
                mask = topic_label_valid==0
            else:
                mask= topic_label_valid>-1

            return mask
        measure_flip_mask = get_measure_flip_mask()
        #Getting the masked label valid
        label_valid = label_valid[measure_flip_mask]
        #Skipping if we dont have enough samples after masking --> Gave NaNs when calculating metrics
        if(label_valid.shape[0]==0):
            return

        #Getting the validation accuracy of the main task
        #TODO: This assumes that this is the last layer of the both branches
        if self.model_args["loss_type"]=="x_entropy":
            #Getting the probability
            main_valid_prob_actual = self.main_task_classifier(X_proj_actual)[measure_flip_mask] #This is softmaxed already
            main_valid_prob_flip = self.main_task_classifier(X_proj_flip)[measure_flip_mask]

            #Getting the difference in the log prob bw the actual and pertured
            main_valid_logprob_actual = tf.math.log(main_valid_prob_actual+self.small_epsilon)
            main_valid_logprob_flip   = tf.math.log(main_valid_prob_flip+self.small_epsilon)
            logprob_delta = tf.math.reduce_mean(tf.math.abs(main_valid_logprob_actual-main_valid_logprob_flip))
            self.topic_flip_main_logprob_delta_list[cidx].update_state(logprob_delta)
        elif self.model_args["loss_type"]=="linear_svm":
            #Getting the actual prob
            main_valid_margin_actual = self.main_task_classifier(X_proj_actual)[measure_flip_mask]
            main_valid_prob_actual = self.get_max_margin_prob_vector(main_valid_margin_actual)[measure_flip_mask]

            #Getting the flip prob
            main_valid_margin_flip = self.main_task_classifier(X_proj_flip)
            main_valid_prob_flip = self.get_max_margin_prob_vector(main_valid_margin_flip)

            #Saving the margin delta in this logprpb delta instead
            logprob_delta = tf.math.reduce_mean(tf.math.abs(main_valid_margin_actual-main_valid_margin_flip))
            self.topic_flip_main_logprob_delta_list[cidx].update_state(logprob_delta)
        else:
            raise NotImplementedError()
        

        #Getting the global probability after pertubation
        self.topic_flip_main_valid_accuracy_list[cidx].update_state(label_valid,main_valid_prob_flip)


        #Now getting the difference in the probability bw actual and perturbed
        p_delta = tf.math.reduce_mean(tf.math.abs(main_valid_prob_actual-main_valid_prob_flip))
        self.topic_flip_main_prob_delta_list[cidx].update_state(p_delta)
 
        #Getting the chnage in the embedding after pertubation
        avg_emb_diff = tf.math.reduce_mean(tf.norm((X_proj_actual-X_proj_flip),axis=-1))
        self.topic_flip_emb_diff_list[cidx].update_state(avg_emb_diff)
 
    def _get_proj_X_enc(self,X_enc,P_matrix):
        '''
        This will give the projected encoded latent representaion which will be furthur
        removed of the information.
        '''
        #Converting to the numpy array
        P_matrix = tf.constant(P_matrix.T,dtype=tf.float32)

        #Now projecting this latent represention into null space
        X_proj = tf.matmul(X_enc,P_matrix)

        return X_proj
    
    def get_angle_between_classifiers(self,class_idx):
        '''
        A utility function that will tell us where the convergence of each of the classifier
        is happening with respect to one another since in high definition it dosent
        make sense to choose any one reference point also.

        Problems:
        1. Here we could have multi-label classifier so we may not have just one direction
            to remove + which direction to take into consideration?
            (Try calculating the angle within the pos+neg class)

        '''
        #Getting the classifier angles
        classifier_dict = {
            "topic{}".format(tidx) : self.topic_task_classifier_list[tidx].get_weights()[0][:,class_idx]
                for tidx in range(self.data_args["num_topics"])
        }
        classifier_dict["main"] = self.main_task_classifier.get_weights()[0][:,class_idx]

        #Getting the norm of the classifier
        classifier_norm = {
            cname:np.linalg.norm(classifier_dict[cname])
                for cname in classifier_dict.keys()
        }
        print("Classifiers Norm:")
        mypp(classifier_norm)

        #Now its time to get the angle among each of them
        convergence_angle=defaultdict(dict)
        for cname1,w1 in classifier_dict.items():
            for cname2,w2 in classifier_dict.items():
                norm1 = np.linalg.norm(w1)
                norm2 = np.linalg.norm(w2)

                angle = np.arccos(np.sum(w1*w2)/(norm1*norm2))/np.pi
                convergence_angle[cname1][cname2]=float(angle)
        
        #Getting the spurous reatio in case we have the dimensions known
        if self.data_args["dtype"]=="toytabular2" and self.model_args["num_hidden_layer"]==0:
            assert classifier_dict["main"].shape[0]==(self.data_args["inv_dims"]+self.data_args["sp_dims"])
            assert len(classifier_dict["main"].shape)==1
            #The first few inv dims are invariant feature
            w_inv = classifier_dict["main"][0:self.data_args["inv_dims"]]
            w_sp  = classifier_dict["main"][self.data_args["inv_dims"]:]

            convergence_angle["w_sp"]["w_inv"]=float(np.linalg.norm(w_sp)/np.linalg.norm(w_inv))


        print("Convergence Angle: (in multiple of pi)")
        mypp(convergence_angle)

        return convergence_angle
    
    def get_all_classifier_accuracy(self,):
        '''
        '''
        classifier_accuracy = {}
        classifier_accuracy["main"]=float(self.main_valid_accuracy.result().numpy())
        for tidx in range(self.data_args["num_topics"]):
            classifier_accuracy["topic{}".format(tidx)]=float(self.topic_valid_accuracy_list[tidx].result().numpy())
            classifier_accuracy["topic{}_flip_main".format(tidx)]=float(self.topic_flip_main_valid_accuracy_list[tidx].result().numpy())
            classifier_accuracy["topic{}_smin_main".format(tidx)]=float(self.topic_smin_main_valid_accuracy_list[tidx].result().numpy())
            classifier_accuracy["topic{}_flip_main_pdelta".format(tidx)]=float(self.topic_flip_main_prob_delta_list[tidx].result().numpy())
            classifier_accuracy["topic{}_flip_main_logpdelta".format(tidx)]=float(self.topic_flip_main_logprob_delta_list[tidx].result().numpy())
            classifier_accuracy["topic{}_flip_emb_diff".format(tidx)]=float(self.topic_flip_emb_diff_list[tidx].result().numpy())

        return classifier_accuracy
    
    def save_data_for_embedding_proj(self,cat_dataset,P_matrix,fname):
        '''
        Convert all the validation dataset into embedding for viz
        '''
        all_input_emb = []
        all_latent_emb = []
        all_tidx_val = []
        for data_batch in cat_dataset:
            label = data_batch["label"]
            idx = data_batch["input_idx"]
            topic_label = data_batch["topic_label"]

            #Taking aside a chunk of data for validation
            valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )
            #Getting the validation data
            label_valid = label[valid_idx:]
            idx_valid = idx[valid_idx:]
            if(idx_valid.shape[0]==0):
                continue
            attn_mask_valid=None
            if self.model_args["bert_as_encoder"]==True:
                attn_mask_valid = data_batch["attn_mask"][valid_idx:]

            #Getting the topic index
            topic_label_valid = topic_label[valid_idx:,:].numpy()
            all_tidx_val.append(topic_label_valid)

            #Now getting the input embedding
            input_emb = self.pre_encoder_layer(idx_valid,attn_mask_valid).numpy()
            all_input_emb.append(input_emb)

            #Now getting the last latent embedding
            X_latent = self._encoder(idx_valid,attn_mask=attn_mask_valid,training=False)
            X_proj = self._get_proj_X_enc(X_latent,P_matrix).numpy()
            all_latent_emb.append(X_proj)
        
        #Consolidating all the data
        all_input_emb = np.concatenate(all_input_emb,axis=0)
        all_latent_emb = np.concatenate(all_latent_emb,axis=0)
        all_tidx_val = np.concatenate(all_tidx_val,axis=0)

        #Saving the embeddings into a text file
        print("Dumping the emb_projector data at: "+fname)
        np.savetxt(fname=fname+"input_emb.txt",X=all_input_emb,delimiter="\t")
        np.savetxt(fname=fname+"latent_emb.txt",X=all_latent_emb,delimiter="\t")
        np.savetxt(
                    fname=fname+"tidx_val.txt",
                    X=all_tidx_val,delimiter="\t",
                    header="\t".join(
                        ["topic_{}".format(tidx) 
                                for tidx in range(self.data_args["num_topics"])
                        ]
                    )
        )
    
    def save_the_probe_metric_list(self):
        '''
        '''
        #Getting the classifier information
        init_conv_angle = self.get_angle_between_classifiers(class_idx=0)
        #Getting the initial classifier accuracy
        init_classifier_acc = self.get_all_classifier_accuracy()
        self.probe_metric_list.append(dict(
                    conv_angle_dict = init_conv_angle,
                    classifier_acc_dict = init_classifier_acc
        ))
        #Dumping the first set of metrics we have
        probe_metric_path = "nlp_logs/{}/probe_metric_list.json".format(data_args["expt_name"])
        print("Dumping the probe metrics in: {}".format(probe_metric_path))
        with open(probe_metric_path,"w") as whandle:
            json.dump(self.probe_metric_list,whandle,indent="\t")
    
    def save_the_experiment_config(self,):
        '''
        This function will save all hte expeirments config into one json file.
        '''
        config={}
        #Adding the data_args
        for key,value in self.data_args.items():
            config[key]=value 
        
        #Getting the model args
        for key,value in self.model_args.items():
            config[key]=value
        
        #Saving the config
        self.config = config
        
        #Saving the config
        config_path = "nlp_logs/{}/config.json".format(data_args["expt_name"])
        print("Dumping the config in: {}".format(config_path))
        with open(config_path,"w") as whandle:
            json.dump(config,whandle,indent="\t")


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -1*dy
    return y, custom_grad

def transformer_trainer_stage1(data_args,model_args):
    '''
    '''
    # #Creating the metadata folder
    # data_args["expt_meta_path"] = "{}/{}".format(
    #                                         data_args["expt_meta_path"],
    #                                         model_args["expt_name"]
    # )
    # os.makedirs(data_args["expt_meta_path"],exist_ok=True)

    #First of all we will load the gate array we are truing to run gate experiemnt
    gate_tensor = get_dimension_gate(data_args,model_args)
    #Saving the gate tensor 
    gate_fpath = "{}/gate_tensor".format(
                                        data_args["expt_meta_path"]
    )
    np.savez(gate_fpath,gate_tensor=gate_tensor.numpy())

    #Dumping the model arguments
    # dump_arguments(data_args,data_args["expt_meta_path"],"data_args")
    # dump_arguments(model_args,data_args["expt_meta_path"],"model_args")

    #First of all creating the model
    classifier = TransformerClassifier(data_args,model_args)

    #Now we will compile the model
    classifier.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )

    #Creating the dataset object
    # data_handler = DataHandleTransformer(data_args)
    # all_cat_ds,all_topic_ds = data_handler.amazon_reviews_handler()
    #Lets reuse the same dataset
    all_cat_ds = data_args["all_cat_ds"]

    # checkpoint_path = "nlp_logs/{}/cp.ckpt".format(model_args["expt_name"])
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                              save_weights_only=True,
    #                                              verbose=1)
    
    #Loading the weight if we want to reuse the computation
    #Switching off this facility for now, will resume later once we are good on toy example
    # if model_args["load_weight_exp"]!=None:
    #     load_path = "nlp_logs/{}/cp_{}.ckpt".format(
    #                                 model_args["load_weight_exp"],
    #                                 model_args["load_weight_epoch"]
    #     )
    #     load_dir = os.path.dirname(load_path)

    #     classifier.load_weights(load_path)


    #Now fitting the model
    # classifier.fit(
    #                     dataset,
    #                     epochs=model_args["epochs"],
    #                     callbacks=[cp_callback]
    #                     # validation_split=model_args["valid_split"]
    # )
    #Writing the custom training loop

    #First of all we have to train our own topic classifier
    # for eidx in range(model_args["epochs"]):
    #     #Now first we will reset all the metrics
    #     classifier.reset_all_metrics()
    #     print("\n==============================================")
    #     print("Starting Epoch: ",eidx)
    #     print("==============================================")

    #     #Now its time to train the topics
    #     for tidx,(tname,topic_ds) in enumerate(all_topic_ds.items()):
    #         #Training the topic through all the batches
    #         for data_batch in topic_ds:
    #             classifier.train_step(tidx,data_batch,gate_tensor,"topic")
            
    #         #Pringitn ghte metrics
    #         print("topic:{}\tceloss:{:0.5f}\tl1_loss:{:0.5f}\tvacc:{:0.5f}".format(
    #                                         tname,
    #                                         classifier.topic_pred_xentropy.result(),
    #                                         classifier.topic_l1_loss_list[tidx].result(),
    #                                         classifier.topic_valid_acc_list[tidx].result(),
    #                 )
    #         )

    #     #Saving the paramenters
    #     checkpoint_path = "nlp_logs/{}/cp_topic_{}.ckpt".format(model_args["expt_name"],eidx)
    #     classifier.save_weights(checkpoint_path)
    

    for eidx in range(model_args["epochs"]):
        #Now first we will reset all the metrics
        classifier.reset_all_metrics()
        print("\n==============================================")
        print("Starting Epoch: ",eidx)
        print("==============================================")


        #Next its time to train the classifier one by one
        for cidx,(cat,cat_ds) in enumerate(all_cat_ds.items()):
            #Trining theis classfifer for full one batch
            for data_batch in cat_ds:
                classifier.train_step_stage1(cidx,data_batch,gate_tensor,"sentiment")

            #Now we will print the metric for this category
            print("cat:{}\tceloss:{:0.5f}\tl1_loss:{:0.5f}\tvacc:{:0.5f}".format(
                                            cat,
                                            classifier.sent_pred_xentropy.result(),
                                            classifier.sent_l1_loss_list[cidx].result(),
                                            classifier.sent_valid_acc_list[cidx].result(),
                    )
            )

        #Saving the paramenters
        checkpoint_path = "{}/cp_cat_{}.ckpt".format(data_args["expt_meta_path"],eidx)
        classifier.save_weights(checkpoint_path)
    
    #Printing the variance of the importance weight
    # get_cat_temb_importance_weight_variance(classifier)
    return

def transformer_trainer_stage2_adversarial(data_args,model_args):
    '''
    This transformer trainer is just for debugging purpose of the model trained on 
    a particular domain/dataset/merged one.

    Right now we suppport the checking the bug for a particular topic within the 
    current model/predictor.
    TODO: We could have multiple bug being checked at a time for the same predictor.
    '''
    #Creating one single dataset for time saving and uniformity
    data_handler = DataHandleTransformer(data_args)
    if "amazon" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.amazon_reviews_handler()
    elif "nlp_toy" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.toy_nlp_dataset_handler()
    else:
        raise NotImplementedError()
    #Getting the dataset for the required category and topic
    print("Getting the dataset for: cat:{}\ttopic:{}".format(
                                                data_args["cat_list"][data_args["debug_cidx"]],
                                                data_args["debug_tidx"],
    ))
    cat_dataset = data_handler._convert_df_to_dataset_stage2_transformer(
                            df=new_all_cat_df[data_args["cat_list"][data_args["debug_cidx"]]],
                            doc_col_name="doc",
                            label_col_name="label",
                            topic_feature_col_name="topic_feature",
                            debug_topic_idx=data_args["debug_tidx"]
    )


    #Step 1: We need to train the optimal classifier without topic removal
    print("Step 1: Training the Main Classifier (to be debugged later)")
    #Creating the forst classifier
    # classifier_main = TransformerClassifier(data_args,model_args)
    # #Now we will compile the model
    # classifier_main.compile(
    #     keras.optimizers.Adam(learning_rate=model_args["lr"])
    # )
    optimal_vacc_main = None
    # for eidx in range(model_args["epochs"]):
    #     classifier_main.reset_all_metrics()
    #     for data_batch in cat_dataset:
    #         classifier_main.train_step_stage2(
    #                                         cidx=data_args["debug_cidx"],
    #                                         tidx=data_args["debug_tidx"],
    #                                         single_ds=data_batch,
    #                                         task="sentiment")

    #     #Now we will print the metric for this category
    #     print("cat:{}\tceloss:{:0.5f}\tl1_loss:{:0.5f}\tvacc:{:0.5f}".format(
    #                         data_args["cat_list"][data_args["debug_cidx"]],
    #                         classifier_main.sent_pred_xentropy.result(),
    #                         classifier_main.sent_l1_loss_list[data_args["debug_cidx"]].result(),
    #                         classifier_main.sent_valid_acc_list[data_args["debug_cidx"]].result(),
    #             )
    #     )
    #     #Keeping track of the optimal vaccuracy of the main classifier
    #     optimal_vacc_main = classifier_main.sent_valid_acc_list[data_args["debug_cidx"]].result()

    #     #Saving the paramenters
    #     checkpoint_path = "{}/cp_cat_main_{}.ckpt".format(data_args["expt_name"],eidx)
    #     classifier_main.save_weights(checkpoint_path)
    


    #Step 2: Now we will remove the topic information and see if performance drops
    print("Step 2: Training the Removal Classifier")
    #Creating the forst classifier
    classifier_trm = TransformerClassifier(data_args,model_args)
    #Now we will compile the model
    classifier_trm.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )
    optimal_vacc_trm = None
    for eidx in range(model_args["epochs"]):
        print("==========================================")
        classifier_trm.reset_all_metrics()
        for data_batch in cat_dataset:
            classifier_trm.train_step_stage2(
                                            cidx=data_args["debug_cidx"],
                                            tidx=data_args["debug_tidx"],
                                            single_ds=data_batch,
                                            task="remove_topic",
                                            reverse_grad=model_args["reverse_grad"]
            )

            #Now we will print the metric for this category
            print("cat:{}\t_sent_celoss:{:0.5f}\ttopic_celoss:{:0.5f}\tsent_vacc:{:0.5f}\ttopic_vacc:{:0.5f}".format(
                                data_args["cat_list"][data_args["debug_cidx"]],
                                classifier_trm.sent_pred_xentropy.result(),
                                classifier_trm.topic_pred_xentropy.result(),
                                classifier_trm.sent_valid_acc_list[data_args["debug_cidx"]].result(),
                                classifier_trm.topic_valid_acc_list[data_args["debug_tidx"]].result(),
                    )
            )
        #Keeping track of the optimal vaccuracy of the main classifier
        optimal_vacc_trm = classifier_trm.sent_valid_acc_list[data_args["debug_cidx"]].result()

        #Saving the paramenters
        checkpoint_path = "{}/cp_cat_trm_{}.ckpt".format(data_args["expt_name"],eidx)
        classifier_trm.save_weights(checkpoint_path)
    

    print("vacc_main:{:0.5f}\nvacc_trm:{:0.5f}".format(optimal_vacc_main,optimal_vacc_trm))

def check_topic_usage_mmd(cat_dataset,classifier_main):
    #initializing the var dict
    mmd_var_dict = {
        "main_valid_prob" : [],
        "main_label"      : [],
        "topic_label"     : []
    }

    #Getting the prediction and the labels
    P_W = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for data_batch in cat_dataset:
        var_dict = classifier_main.valid_step_stage2(
                                    dataset_batch=data_batch,
                                    P_matrix=P_W
        )
        for key in mmd_var_dict.keys():
            mmd_var_dict[key].append(var_dict[key])
    
    #Consolidating all te variables
    for key in mmd_var_dict.keys():
        mmd_var_dict[key] = np.concatenate(mmd_var_dict[key],axis=0)
    
    #Now we could get the main distribution and conditional distribution
    mmd_var_dict["main_valid_pred"] = np.argmax(mmd_var_dict["main_valid_prob"],axis=-1)

    #Now getting the marginal distribution
    marg_dist_dict = {}
    marg_dist_z_dict = {}
    dist_matrix_yz = np.zeros((4,2),dtype=np.float32)
    dist_matrix_z = np.zeros((2,2),dtype=np.float32)
    for yval in [0,1]:
        for zval in [0,1]:
            #Now segmenting the results
            ymask = mmd_var_dict["main_label"]==yval 
            zmask = mmd_var_dict["topic_label"]==zval
            overall_mask = np.logical_and(ymask,zmask)
            #Getting the prediction in this segment
            pred = mmd_var_dict["main_valid_pred"][overall_mask]
            p1 = np.sum(pred)/(len(pred)+1e-15) #if len(pred)!=0 else 0.5
            print("y={},z={},len:{}".format(yval,zval,len(pred)))

            #Getting the marginal distribution for the 
            marg_dist_dict["y={},z={}".format(yval,zval)]=[
                                                    "p(f(x)=0)={:0.2f}".format(1-p1),
                                                    "p(f(x)=1)={:0.2f}".format(p1)
            ]
            dist_matrix_yz[2*yval+zval,0]=1-p1
            dist_matrix_yz[2*yval+zval,1]=p1
            
            #Just getting marginal in z
            pred_z = mmd_var_dict["main_valid_pred"][zmask]
            p1_z = np.sum(pred_z)/(len(pred_z)+1e-15) #if len(pred_z)!=0 else 0.5
            print("z={},len:{}".format(zval,len(pred_z)))
            marg_dist_z_dict["z={}".format(zval)]=[
                                                    "p(f(x)=0)={:0.2f}".format(1-p1_z),
                                                    "p(f(x)=1)={:0.2f}".format(p1_z)
            ]
            dist_matrix_z[zval,0]=1-p1_z
            dist_matrix_z[zval,1]=p1_z
            
    print("Conditional Distribution:")
    mypp(marg_dist_dict)
    print("Conditional Distribution: Z")
    mypp(marg_dist_z_dict)
    
    causal_distance = (distance.jensenshannon(dist_matrix_yz[0,:],dist_matrix_yz[1,:])
                       +distance.jensenshannon(dist_matrix_yz[2,:],dist_matrix_yz[3,:]))
    anticausal_distance = distance.jensenshannon(dist_matrix_z[0,:],dist_matrix_z[1,:])
    print("js_yz:{:0.5f}\n js_z :{:0.5f}".format(causal_distance,anticausal_distance))

def transformer_trainer_stage2_inlp(data_args,model_args):
    '''
    This trainer will train the transformer and then remove the topic using the
    null space projection method.
    '''
    #Creating one single dataset for time saving and uniformity
    data_handler = DataHandleTransformer(data_args)
    if "amazon" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.amazon_reviews_handler()
    elif "nlp_toy" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.toy_nlp_dataset_handler()
    else:
        raise NotImplementedError()
    #Getting the dataset for the required category and topic
    print("Getting the dataset for: cat:{}".format(
                                                data_args["cat_list"][data_args["debug_cidx"]],
    ))
    cat_dataset = data_handler._convert_df_to_dataset_stage2_transformer(
                            df=new_all_cat_df[data_args["cat_list"][data_args["debug_cidx"]]],
                            doc_col_name="doc",
                            label_col_name="label",
                            topic_feature_col_name="topic_feature",
                            debug_topic_idx=None, #data_args["debug_tidx"]#redundant now all topics are there
    )
    data_args["num_topics"]=data_handler.data_args["num_topics"]




    # Step 1: Now we will start training the main task classifier
    print("Stage 1: Training the main predictor (to be debugged later)")
    classifier_main = TransformerClassifier(data_args,model_args)
    #Now we will compile the model
    classifier_main.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )
    #Lets now take out the weights of the of the topic classifier
    topic_classifier_init_weight_list = [] #not initialized here.

    optimal_vacc_main = None
    #Initializing the Identity P-matrix
    P_identity = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for eidx in range(model_args["epochs"]):
        print("============================================")
        classifier_main.reset_all_metrics()
        #Training the main classifier
        for bidx,dataset_batch in enumerate(cat_dataset):
            classifier_main.train_step_stage2_inlp(
                                bidx=bidx,
                                dataset_batch=dataset_batch,
                                task="main",
                                P_matrix=P_identity,
                                cidx=data_args["debug_cidx"],
                                tidx=None,
            )
        
        #Training the topic classifier
        for bidx,dataset_batch in enumerate(cat_dataset):
            for tidx in range(data_args["num_topics"]):
                classifier_main.train_step_stage2_inlp(
                    bidx=bidx,
                    dataset_batch=dataset_batch,
                    task="topic",
                    P_matrix=P_identity,
                    cidx=data_args["debug_cidx"],
                    tidx=tidx,
                )
        
            #Since the weights are initialized when needed hence we need to caputre this here
            if(len(topic_classifier_init_weight_list)==0):
                topic_classifier_init_weight_list = [
                    classifier_main.topic_classifier_list[tidx].get_weights()
                        for tidx in range(data_args["num_topics"])
                ]
        
        #Printing the classifier loss and accuracy
        log_format="epoch:{:}\tcname:{}\txloss:{:0.4f}\tvacc:{:0.3f}"
        print(log_format.format(
                            eidx,
                            "main",
                            classifier_main.sent_pred_xentropy_list[data_args["debug_cidx"]].result(),
                            classifier_main.sent_valid_acc_list[data_args["debug_cidx"]].result(),
        ))
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            eidx,
                            "topic-{}".format(tidx),
                            classifier_main.topic_pred_xentropy_list[tidx].result(),
                            classifier_main.topic_valid_acc_list[tidx].result()
            ))

        #Keeping track of the optimal vaccuracy of the main classifier
        optimal_vacc_main = classifier_main.sent_valid_acc_list[data_args["debug_cidx"]].result()
    
    #Metrics to probe the classifiers
    probe_metric_list_actual = [] #list of  (conv_angle_dict,accracy_dict)
    #Getting the classifier information
    init_conv_angle = classifier_main.get_angle_between_classifiers(class_idx=0)
    #Getting the initial classifier accuracy
    init_classifier_acc = classifier_main.get_all_classifier_accuracy()
    probe_metric_list_actual.append(dict(
                conv_angle_dict = init_conv_angle,
                classifier_acc_dict = init_classifier_acc
    ))









    #Step 2: Next we will start the removal process
    #Next we will be going to use this trained classifier to do the null space projection
    for didx in range(data_args["num_topics"]):
        #Updating the debug tidx
        '''
        The only problem with this iterating is that the first removal will have 
        better trained topic classifier than the others.
        '''
        data_args["debug_tidx"] = didx 
        data_handler.data_args["debug_tidx"] = didx
        #Giving it a separate probe metric list
        probe_metric_list = probe_metric_list_actual.copy() 


        print("\n\nStage 2: Removing the topic information for topic:{}".format(data_args["debug_tidx"]))
        all_proj_matrix_list = []
        P_W = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
        for pidx in range(model_args["num_proj_iter"]):
            #Resetting all the metrics
            classifier_main.reset_all_metrics()

            #Now resetting the topic classifier weights again
            for tidx in range(data_args["num_topics"]):
                classifier_main.topic_classifier_list[tidx].set_weights(
                                    topic_classifier_init_weight_list[tidx]
                )

            for tidx in range(data_args["num_topics"]):
                #Step1: Training the topic classifier now
                tbar = tqdm(range(model_args["topic_epochs"]))
                #Resetting the topic classifier
                # classifier_main.topic_task_classifier = layers.Dense(2,activation="softmax")
                for eidx in tbar:
                    for bidx,data_batch in enumerate(cat_dataset):
                        classifier_main.train_step_stage2_inlp(
                                            bidx=bidx,
                                            dataset_batch=data_batch,
                                            task="topic_inlp",
                                            P_matrix=P_W,
                                            cidx=data_args["debug_cidx"],
                                            tidx=tidx,
                        )

                    #Updating the description of the tqdm
                    tbar.set_postfix_str("tceloss:{:0.4f},  tvacc:{:0.3f}".format(
                                                classifier_main.topic_pred_xentropy_list[tidx].result(),
                                                classifier_main.topic_valid_acc_list[tidx].result()
                    ))
            #Get the topic metrics aftertraining the classifier
            topic_vacc_before = classifier_main.topic_valid_acc_list[data_args["debug_tidx"]].result()

            #Getting the classifiers angle (after this step of training)
            classifier_main.reset_all_metrics()
            for bidx,data_batch in enumerate(cat_dataset):
                for tidx in range(data_args["num_topics"]):
                    classifier_main.valid_step_stage2_inlp(
                                                bidx=bidx,
                                                dataset_batch=data_batch,
                                                P_matrix=P_W,
                                                cidx=data_args["debug_cidx"],
                                                tidx=tidx,
                    )
            
            #Getting the classifier information
            conv_angle_dict = classifier_main.get_angle_between_classifiers(class_idx=0)
            classifier_acc_dict = classifier_main.get_all_classifier_accuracy()
            probe_metric_list.append(dict(
                        conv_angle_dict=conv_angle_dict,
                        classifier_acc_dict=classifier_acc_dict,
            ))

            #Step2: Remove the information and get the projection
            topic_W_matrix = classifier_main.topic_classifier_list[data_args["debug_tidx"]].get_weights()[0].T
            #Now getting the projection matrix
            P_W_curr = get_rowspace_projection(topic_W_matrix)
            all_proj_matrix_list.append(P_W_curr)
            #Getting the aggregate projection matrix
            P_W = get_projection_to_intersection_of_nullspaces(
                                            rowspace_projection_matrices=all_proj_matrix_list,
                                            input_dim=classifier_main.hlayer_dim
            )


            #Step 3: Now we need to retrain the main task head, 
            #to again get optimal classifier in this projected space
            #This seems like not a tright thing to do, as mechanisam change could happen and 
            #we could get equaivalent performance from other feature, though the spurious was being used.
            # cbar = tqdm(range(model_args["topic_epochs"]))
            # for eidx in cbar:
            #     for data_batch in cat_dataset:
            #         classifier_main.train_step_stage2(
            #                             dataset_batch=data_batch,
            #                             task="inlp_main",
            #                             P_matrix=P_W,
            #         )

            #     #Updating the description of the tqdm
            #     cbar.set_postfix_str("mceloss:{:0.4f},  mvacc:{:0.3f}".format(
            #                                 classifier_main.main_pred_xentropy.result(),
            #                                 classifier_main.main_valid_accuracy.result()
            #     ))


            #Now we need to get the validation accuracy on this projected matrix
            classifier_main.reset_all_metrics()
            for bidx,data_batch in enumerate(cat_dataset):
                for tidx in range(data_args["num_topics"]):
                    classifier_main.valid_step_stage2_inlp(
                                                bidx=bidx,
                                                dataset_batch=data_batch,
                                                P_matrix=P_W,
                                                cidx=data_args["debug_cidx"],
                                                tidx=tidx,
                    )
            
            print("pidx:{:}\tmain_init:{:0.3f}\tmain_after:{:0.3f}\ttopic_before:{:0.3f}\ttopic_after:{:0.3f}".format(
                                                pidx,
                                                optimal_vacc_main,
                                                classifier_main.sent_valid_acc_list[data_args["debug_cidx"]].result(),
                                                topic_vacc_before,
                                                classifier_main.topic_valid_acc_list[data_args["debug_tidx"]].result()
            ))
        
            #Saving the probe metrics in a json file 
            probe_metric_path = "nlp_logs/{}/".format(data_args["expt_name"])
            pathlib.Path(probe_metric_path).mkdir(parents=True, exist_ok=True)
            print("Dumping the probe metrics in: {}".format(probe_metric_path))
            probe_metric_fname = probe_metric_path+"probe_metric_list_topic_{}.json".format(data_args["debug_tidx"])
            with open(probe_metric_fname,"w") as whandle:
                json.dump(probe_metric_list,whandle,indent="\t")





def nbow_trainer_stage2(data_args,model_args):
    '''
    This will train neural-BOW model for the stage 2.
    '''
    #Creating one single dataset for time saving and uniformity
    data_handler = DataHandleTransformer(data_args)
    if "amazon" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.amazon_reviews_handler()
    elif "nlp_toy2" in data_args["path"]:
        if data_args["dtype"]=="toynlp2":
            cat_dataset = data_handler.toy_nlp_dataset_handler2()
        elif data_args["dtype"]=="toytabular2":
            cat_dataset = data_handler.toy_tabular_dataset_handler2()
    elif "nlp_toy" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.toy_nlp_dataset_handler()
    elif "multinli" in data_args["path"]:
        cat_dataset = data_handler.controlled_multinli_dataset_handler()
    else:
        raise NotImplementedError()
    
    if "nlp_toy2" not in data_args["path"] and "multinli" not in data_args["path"]:
        #NLP TOY2 has its data processing already bulit in its generation function
        #Getting the dataset for the required category and topic
        print("Getting the dataset for: cat:{}\ttopic:{}".format(
                                                    data_args["cat_list"][data_args["debug_cidx"]],
                                                    data_args["debug_tidx"],
        ))
        cat_dataset = data_handler._convert_df_to_dataset_stage2_NBOW(
                                df=new_all_cat_df[data_args["cat_list"][data_args["debug_cidx"]]],
                                pdoc_col_name="pdoc",
                                label_col_name="label",
                                topic_feature_col_name="topic_feature",
                                debug_topic_idx=data_args["debug_tidx"]
        )

        #Updating the number of topics
        data_args["num_topics"]=len(data_handler.topic_list)



    #Creating the forst classifier
    classifier_main = SimpleNBOW(data_args,model_args,data_handler)
    #Now we will compile the model
    classifier_main.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )
    optimal_vacc_main = None


    print("Stage 1: Training the main classifier! (to be debugged later)")
    if(model_args["main_model_mode"]=="causal" and (
                                        "nlp_toy2" in data_args["path"] or
                                        "multinli" in data_args["path"]
                                    )
    ):
        #Here we will train on the main classifier to be purely causal
        print("Training a pure causal classifier where there is no spurious correlation")

        #building the dataset where the spurious feature has p=0.5
        if "multinli" in data_args["path"]:
            init_tneg_pval = data_handler.data_args["neg_topic_corr"]
            data_handler.data_args["neg_topic_corr"]=0.5
        else:
            init_t1_pval = data_handler.data_args["topic_corr_list"][-1]
            data_handler.data_args["topic_corr_list"][-1]=0.5
        
        if data_args["dtype"]=="toynlp2":
            causal_cat_dataset = data_handler.toy_nlp_dataset_handler2()
        elif data_args["dtype"]=="toytabular2":
            causal_cat_dataset = data_handler.toy_tabular_dataset_handler2()
        elif "multinli" in data_args["path"]:
            causal_cat_dataset = data_handler.controlled_multinli_dataset_handler()
        
        if "multinli" in data_args["path"]:
            data_handler.data_args["neg_topic_corr"]=init_tneg_pval
        else:
            data_handler.data_args["topic_corr_list"][-1]=init_t1_pval

        #Now training the main classifier on this causal dataset
        classifier_main = perform_main_classifier_training(
                                        data_args,
                                        model_args,
                                        causal_cat_dataset,
                                        classifier_main
        )

    elif(model_args["main_model_mode"]=="non_causal"):
        #Run the training normally
        classifier_main = perform_main_classifier_training(
                                        data_args,
                                        model_args,
                                        cat_dataset,
                                        classifier_main
        )

        #Retraining the head of the classifier only given encoder is warm
        if(model_args["head_retrain_mode"]=="warm_encoder"):
            print("\n\n\nRetraining all the heads with the warmed up encoder!")
            classifier_main = perform_head_classifier_training(
                                        data_args,
                                        model_args,
                                        cat_dataset,
                                        classifier_main,
            )
        
        #Saving the embedding for visaulization
        if data_args["save_emb"]==True:
            emb_savepath = "nlp_logs/{}/".format(data_args["expt_name"])
            classifier_main.save_data_for_embedding_proj(
                    cat_dataset=cat_dataset,
                    P_matrix=np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim),
                    fname=emb_savepath,
            )
    else:
        raise NotImplementedError()

    #Getting the optimal validation accuracy of main classifier
    optimal_vacc_main = classifier_main.main_valid_accuracy.result()
    
    #Wont be using the removal part right now.
    # sys.exit(0)
    #Getting the MMD metrics to see usage
    # check_topic_usage_mmd(cat_dataset,classifier_main)


    ##################################################################
    # ADVERSARIAL REMOVAL
    ##################################################################
    if(model_args["removal_mode"]=="adversarial"):
        print("Stage 2: Removing the topic information adversarially!")
        perform_adversarial_removal_nbow(
                                        cat_dataset=cat_dataset,
                                        classifier_main=classifier_main,
        )
    elif(model_args["removal_mode"]=="null_space"):
        print("Stage 2: Removing the topic information!")
        perform_null_space_removal_nbow(
                                        cat_dataset=cat_dataset,
                                        classifier_main=classifier_main,
                                        optimal_vacc_main=optimal_vacc_main,
        )
    
    return

def perform_head_classifier_training(data_args,model_args,cat_dataset,classifier_main):
    '''
    This function will be used to retrain the main head again after we have primed
    the encoder with the reavant topic information presence.

    In this training the main head's parameter will be randomized and retrained
    keeping the encoder fixed.


    Lets retrain the other topic heads also when we are sure that topic info has
    come to the top layer atleast.
    '''
    #First of all we have to randomize all the heads again
    classifier_main.set_all_head_init_weights()

    #Now restarting our training again
    P_identity = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for eidx in range(model_args["epochs"]):
        print("==========================================")
        classifier_main.reset_all_metrics()
        for data_batch in cat_dataset:
            #Training the main task classifier
            classifier_main.train_step_stage2(
                                    dataset_batch=data_batch,
                                    task="inlp_main",
                                    P_matrix=P_identity,
                                    cidx=None,
            )
            #Training the topics too to ensure that topic information is there
            #TODO: Here dont give task as inlp topic as they dont train the encoder. Create new task
            for tidx in range(data_args["num_topics"]):
                classifier_main.train_step_stage2(
                                        dataset_batch=data_batch,
                                        task="inlp_topic",
                                        P_matrix=P_identity,
                                        cidx=tidx
                )
                if model_args["measure_flip_pdelta"]==True:
                    #Updating the flip topic main accuracy also
                    classifier_main.valid_step_stage2_flip_topic(
                                            dataset_batch=data_batch,
                                            P_matrix=P_identity,
                                            cidx=tidx,
                    )
        
        #Printing the classifier loss and accuracy
        log_format="epoch:{:}\tcname:{}\txloss:{:0.4f}\tvacc:{:0.3f}"
        print(log_format.format(
                            eidx,
                            "main",
                            classifier_main.main_pred_xentropy.result(),
                            classifier_main.main_valid_accuracy.result(),
        ))
        #Printing the topic accuracy
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            eidx,
                            "topic-{}".format(tidx),
                            classifier_main.topic_pred_xentropy_list[tidx].result(),
                            classifier_main.topic_valid_accuracy_list[tidx].result()
            ))
        
        #Printing the flip-topic main accuracy (proxy for feature use coefficient)
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            eidx,
                            "topic-{}_flip_main".format(tidx),
                            classifier_main.topic_flip_main_prob_delta_list[tidx].result(),
                            classifier_main.topic_flip_main_valid_accuracy_list[tidx].result()
            ))
        
        #Keeping track of the optimal vaccuracy of the main classifier
        optimal_vacc_main = classifier_main.main_valid_accuracy.result()

        #Saving the paramenters
        # checkpoint_path = "nlp_logs/{}/cp_cat_main_{}.ckpt".format(data_args["expt_name"],eidx)
        # classifier_main.save_weights(checkpoint_path)

        #Saving the proble metric list
        classifier_main.save_the_probe_metric_list()
    
    return classifier_main

def perform_main_classifier_training(data_args,model_args,cat_dataset,classifier_main):
    '''
    '''
    P_identity = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for eidx in range(model_args["epochs"]):
        print("==========================================")
        classifier_main.reset_all_metrics()
        tbar = tqdm(range(len(cat_dataset)))
        for bidx,data_batch in zip(tbar,cat_dataset):
            tbar.set_postfix_str("Batch:{}  bidx:{}".format(len(cat_dataset),bidx))
            #Training the main task classifier
            classifier_main.train_step_stage2(
                                    dataset_batch=data_batch,
                                    task="main",
                                    P_matrix=P_identity,
                                    cidx=None,
            )
            #Training the topics too to ensure that topic information is there
            #TODO: Here dont give task as inlp topic as they dont train the encoder. Create new task
            for tidx in range(data_args["num_topics"]):
                classifier_main.train_step_stage2(
                                        dataset_batch=data_batch,
                                        task="topic",
                                        P_matrix=P_identity,
                                        cidx=tidx
                )
            
            #Here we will capture the model weights for re-init later
            classifier_main.get_all_head_init_weights()
        
        #Updating the validation acccuracy of the model
        for data_batch in cat_dataset:
            for tidx in range(data_args["num_topics"]):
                #Updating the topic accuracies and the main accuracies
                classifier_main.valid_step_stage2(
                                        dataset_batch=data_batch,
                                        P_matrix=P_identity,
                                        cidx=tidx,
                )

                if model_args["measure_flip_pdelta"]==True:
                    #Updating the flip topic main accuracy also
                    classifier_main.valid_step_stage2_flip_topic(
                                            dataset_batch=data_batch,
                                            P_matrix=P_identity,
                                            cidx=tidx,
                    )
        
        #Printing the classifier loss and accuracy
        log_format="epoch:{:}\tcname:{}\txloss:{:0.4f}\tvacc:{:0.3f}"
        print(log_format.format(
                            eidx,
                            "main",
                            classifier_main.main_pred_xentropy.result(),
                            classifier_main.main_valid_accuracy.result(),
        ))
        #Printing the topic accuracy
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            eidx,
                            "topic-{}".format(tidx),
                            classifier_main.topic_pred_xentropy_list[tidx].result(),
                            classifier_main.topic_valid_accuracy_list[tidx].result()
            ))
        
        #Printing the flip-topic main accuracy (proxy for feature use coefficient)
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            eidx,
                            "topic-{}_flip_main".format(tidx),
                            classifier_main.topic_flip_main_prob_delta_list[tidx].result(),
                            classifier_main.topic_flip_main_valid_accuracy_list[tidx].result()
            ))
        
        #Keeping track of the optimal vaccuracy of the main classifier
        optimal_vacc_main = classifier_main.main_valid_accuracy.result()

        #Saving the paramenters
        # checkpoint_path = "nlp_logs/{}/cp_cat_main_{}.ckpt".format(data_args["expt_name"],eidx)
        # classifier_main.save_weights(checkpoint_path)

        #Saving the probe metric list
        classifier_main.save_the_probe_metric_list()
    
    return classifier_main

def perform_adversarial_removal_nbow(cat_dataset,classifier_main):
    '''
    '''
    P_identity = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for ridx in range(model_args["adv_rm_epochs"]):
        print("==========================================")
        classifier_main.reset_all_metrics()
        tbar = tqdm(range(len(cat_dataset)))
        for bidx,data_batch in zip(tbar,cat_dataset):
            tbar.set_postfix_str("Batch:{}  bidx:{}".format(len(cat_dataset),bidx))

            if model_args["adv_rm_method"]=="adv_rm_only_topic":
                #Training the main task classifier
                classifier_main.train_step_stage2(
                                        dataset_batch=data_batch,
                                        task="main",
                                        P_matrix=P_identity,
                                        cidx=None,
                )
                #Remove the debug_idx topic from the latent space adversarially
                classifier_main.train_step_stage2(
                                                dataset_batch=data_batch,
                                                task="adv_rm_only_topic",
                                                P_matrix=P_identity,
                                                cidx=data_args["debug_tidx"],
                )
            elif model_args["adv_rm_method"]=="adv_rm_with_main":
                classifier_main.train_step_stage2(
                                        dataset_batch=data_batch,
                                        task="adv_rm_with_main",
                                        P_matrix=P_identity,
                                        cidx=None,
                                        adv_rm_tidx=data_args["debug_tidx"],
                )
            else:
                raise NotImplementedError()

        #Getting the validation scores once the train step is complete
        for data_batch in cat_dataset:
            #Validate the other topic so that we know the accuracy etc
            for tidx in range(data_args["num_topics"]):
                #Getting the validation accuracy
                classifier_main.valid_step_stage2(
                                    dataset_batch=data_batch,
                                    P_matrix=P_identity,
                                    cidx=tidx
                )
            
            #In real model we dont have the pertubation power
            if model_args["measure_flip_pdelta"]==True:
                #Also let us get the effect of pertubation of the feature on main classifier
                for tidx in range(data_args["num_topics"]):
                    classifier_main.valid_step_stage2_flip_topic(
                                        dataset_batch=data_batch,
                                        P_matrix=P_identity,
                                        cidx=tidx,
                    )
        
        #Printing the classifier loss and accuracy
        log_format="epoch:{:}\tcname:{}\txloss:{:0.4f}\tvacc:{:0.3f}"
        print(log_format.format(
                            ridx,
                            "main",
                            classifier_main.main_pred_xentropy.result(),
                            classifier_main.main_valid_accuracy.result(),
        ))
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            ridx,
                            "topic-{}".format(tidx),
                            classifier_main.topic_pred_xentropy_list[tidx].result(),
                            classifier_main.topic_valid_accuracy_list[tidx].result()
            ))
        for tidx in range(data_args["num_topics"]):
            print(log_format.format(
                            ridx,
                            "topic-{}_flip_main".format(tidx),
                            classifier_main.topic_flip_main_prob_delta_list[tidx].result(),
                            classifier_main.topic_flip_main_valid_accuracy_list[tidx].result()
            ))
        

        #Dumping the probe metrics at every iteration
        classifier_main.save_the_probe_metric_list()

def perform_null_space_removal_nbow(cat_dataset,classifier_main,optimal_vacc_main):
    '''
    '''
    #Next we will be going to use this trained classifier to do the null space projection
    all_proj_matrix_list = []
    P_W = np.eye(classifier_main.hlayer_dim,classifier_main.hlayer_dim)
    for pidx in range(model_args["num_proj_iter"]):
        #Resetting all the metrics
        classifier_main.reset_all_metrics()

        for tidx in range(data_args["num_topics"]):
            #Step1: Training the topic classifier now
            #Resetting the topic classifier
            # classifier_main.topic_task_classifier = layers.Dense(2,activation="softmax")
            if model_args["bert_as_encoder"]==True:
                for tepoch in range(model_args["topic_epochs"]):
                    bbar = tqdm(range(len(cat_dataset)))
                    #Training the model 
                    for bidx,data_batch in zip(bbar,cat_dataset):
                        bbar.set_postfix_str("bsize:{},  bidx:{}".format(len(cat_dataset),bidx))
                        classifier_main.train_step_stage2(
                                            dataset_batch=data_batch,
                                            task="inlp_topic",
                                            P_matrix=P_W,
                                            cidx=tidx,
                        )
                    
                    #Getting the validation accuracy
                    for data_batch in cat_dataset:
                        classifier_main.valid_step_stage2(
                                                dataset_batch=data_batch,
                                                P_matrix=P_W,
                                                cidx=tidx,
                        )
                    print("topic:{:}\ttceloss:{:0.4f}\ttvacc:{:0.3f}".format(
                                                tidx,
                                                classifier_main.topic_pred_xentropy_list[tidx].result(),
                                                classifier_main.topic_valid_accuracy_list[tidx].result()
                    ))

            else:
                tbar = tqdm(range(model_args["topic_epochs"]))
                for eidx in tbar:
                    for data_batch in cat_dataset:
                        classifier_main.train_step_stage2(
                                            dataset_batch=data_batch,
                                            task="inlp_topic",
                                            P_matrix=P_W,
                                            cidx=tidx,
                        )
                    
                    #Getting the validation accuracy
                    for data_batch in cat_dataset:
                        classifier_main.valid_step_stage2(
                                                dataset_batch=data_batch,
                                                P_matrix=P_W,
                                                cidx=tidx,
                        )

                    #Updating the description of the tqdm
                    tbar.set_postfix_str("tceloss:{:0.4f},  tvacc:{:0.3f}".format(
                                                classifier_main.topic_pred_xentropy_list[tidx].result(),
                                                classifier_main.topic_valid_accuracy_list[tidx].result()
                    ))

        #Getting the classifiers angle (after this step of training)
        classifier_main.reset_all_metrics()
        for data_batch in cat_dataset:
            for tidx in range(data_args["num_topics"]):
                #Getting the topic and  validation accuracy
                classifier_main.valid_step_stage2(
                                            dataset_batch=data_batch,
                                            P_matrix=P_W,
                                            cidx=tidx,
                )

                #Getting the validation accuracy when we flip the topic info in input
                if model_args["measure_flip_pdelta"]==True:
                    classifier_main.valid_step_stage2_flip_topic(
                                        dataset_batch=data_batch,
                                        P_matrix=P_W,
                                        cidx=tidx,
                    )
        #Get the topic metrics aftertraining the classifier
        topic_vacc_before = classifier_main.topic_valid_accuracy_list[data_args["debug_tidx"]].result()
        
        #Getting the classifier information
        classifier_main.save_the_probe_metric_list()

        #Step2: Remove the information and get the projection
        topic_W_matrix = classifier_main.topic_task_classifier_list[data_args["debug_tidx"]].get_weights()[0].T
        #Now getting the projection matrix
        P_W_curr = get_rowspace_projection(topic_W_matrix)
        all_proj_matrix_list.append(P_W_curr)
        #Getting the aggregate projection matrix
        P_W = get_projection_to_intersection_of_nullspaces(
                                        rowspace_projection_matrices=all_proj_matrix_list,
                                        input_dim=classifier_main.hlayer_dim
        )


        #Step 3: Now we need to retrain the main task head, 
        #to again get optimal classifier in this projected space
        #This seems like not a tright thing to do, as mechanisam change could happen and 
        #we could get equaivalent performance from other feature, though the spurious was being used.
        # cbar = tqdm(range(model_args["topic_epochs"]))
        # for eidx in cbar:
        #     for data_batch in cat_dataset:
        #         classifier_main.train_step_stage2(
        #                             dataset_batch=data_batch,
        #                             task="inlp_main",
        #                             P_matrix=P_W,
        #         )

        #     #Updating the description of the tqdm
        #     cbar.set_postfix_str("mceloss:{:0.4f},  mvacc:{:0.3f}".format(
        #                                 classifier_main.main_pred_xentropy.result(),
        #                                 classifier_main.main_valid_accuracy.result()
        #     ))


        #Now we need to get the validation accuracy on this projected matrix
        classifier_main.reset_all_metrics()
        for data_batch in cat_dataset:
            for tidx in range(data_args["num_topics"]):
                classifier_main.valid_step_stage2(
                                            dataset_batch=data_batch,
                                            P_matrix=P_W,
                                            cidx=tidx,
                )
        
        print("pidx:{:}\tmain_init:{:0.3f}\tmain_after:{:0.3f}\ttopic_before:{:0.3f}\ttopic_after:{:0.3f}".format(
                                            pidx,
                                            optimal_vacc_main,
                                            classifier_main.main_valid_accuracy.result(),
                                            topic_vacc_before,
                                            classifier_main.topic_valid_accuracy_list[data_args["debug_tidx"]].result()
        ))

def get_cat_temb_importance_weight_variance(classifier):
    #Getting the topic importance
    cat_topic_imp_weights = [
        np.squeeze(tf.sigmoid(classifier.cat_temb_importance_weight_list[cidx]).numpy())
            for cidx in range(len(classifier.data_args["cat_list"]))
    ]

    #Getting the topic importance score
    cat_topic_imp_weights = np.stack(cat_topic_imp_weights,axis=-1)
    cat_topic_imp_mean = np.mean(cat_topic_imp_weights,axis=-1)
    cat_topic_imp_std  = np.std(cat_topic_imp_weights,axis=-1)
    topic_metric_list=[]
    for tidx in range(len(classifier.data_args["topic_list"])):
        print("topic:{}\tmean:{:0.5f}\tstd:{:0.5f}".format(
                                            tidx,
                                            cat_topic_imp_mean[tidx],
                                            cat_topic_imp_std[tidx],
            )
        )

        topic_metric_list.append((tidx,cat_topic_imp_mean[tidx],cat_topic_imp_std[tidx]))
    
    topic_metric_list.sort(key=lambda x:x[-1])
    mypp(topic_metric_list)

def get_sent_imp_weights(classifier):
    '''
    This function will give us the importance score for each of the dimensions
    of the task classifier.
    '''
    sent_weights=[
        np.squeeze(tf.sigmoid(classifier.cat_importance_weight_list[cidx]).numpy())
                        for cidx in range(len(classifier.data_args["cat_list"]))
    ]
    #Getting the flow based weights
    new_sent_weights = []
    for cidx in range(len(classifier.data_args["cat_list"])):
        cimp_weight = np.expand_dims(sent_weights[cidx],axis=0)
        cfirst_weight = np.abs(classifier.cat_classifier_list[cidx].get_weights()[0].T)
        # cfirst_weight = classifier.cat_classifier_list[cidx][0].get_weights()[0].T
        # print("imp:{}\tfirst:{}".format(cimp_weight.shape,cfirst_weight.shape))
        
        new_cimp_weight = np.sum(cfirst_weight*cimp_weight,axis=0)
        new_sent_weights.append(new_cimp_weight)
    
    sent_weights = new_sent_weights
    sent_weights = np.stack(sent_weights,axis=1)

    return sent_weights

def get_topic_imp_weights(classifier):
    raise NotImplementedError()

def get_feature_spuriousness(classifier,ood_vacc,sent_weights):
    '''
    This function will use the drop in the performance idea to get the dimension 
    which are spirous i.e whose importance weiths saw a net negative decrease
    in the importance weight.
    '''
    global_imp_diff = np.zeros(sent_weights.shape[0])
    cat_imp_diff_dict = {}
    for cidx,cat in enumerate(classifier.data_args["cat_list"]):
        cat_vacc = ood_vacc[cat][cat]
        #First we will get the weighted drop in imp from this domain
        cat_imp_diff = np.zeros(sent_weights.shape[0])
        for nidx,nat in enumerate(classifier.data_args["cat_list"]):
            nat_vacc = ood_vacc[cat][nat]
            vacc_delta = nat_vacc - cat_vacc 

            #Skipping if the there is not drop in perf or same category as nat
            if vacc_delta>=0 or cat==nat:
                continue

            #Now we need to get the drop in performance
            temp_imp_diff = np.abs(vacc_delta)\
                                *(sent_weights[:,nidx]-sent_weights[:,cidx])
            #Now we dont know the positive drops are spurious or causal so remove them
            temp_imp_diff = temp_imp_diff*(temp_imp_diff<=0.0)

            #Now adding the contribution to the 
            cat_imp_diff += temp_imp_diff
        
        #Getting the cat specific imp diff for analysis
        cat_imp_diff_dict[cat]=cat_imp_diff
        
        #Adding the differnet to the global diff
        global_imp_diff += cat_imp_diff
    
    #Now we have the effective drop in the importance
    threshold_criteria=model_args["gate_var_cutoff"]
    if threshold_criteria=="neg":
        cutoff_value=0.0
        gate_arr = (global_imp_diff>=0.0)*1.0 #they will become 1 and rest as zero i.e gated
    else:
        #Getting the top negetive ones
        zero_upto = int(float(threshold_criteria)*global_imp_diff.shape[0])
        cutoff_value = np.sort(global_imp_diff)[zero_upto]

        gate_arr = (global_imp_diff>cutoff_value)*1.0

        #Another stratgy could be not to make the gate binary, rather sigmoid(imp_diff)
    
    #Creating the final gate array
    gate_arr = np.expand_dims(gate_arr,axis=0)
    print("cutoff_percent:{}\tcutoff_val:{}\tnum_alive:{}".format(
                                model_args["gate_var_cutoff"],
                                cutoff_value,
                                np.sum(gate_arr)
    ))

    #We can directly print the correlation of the 
    print("Correlation of gate array and cat_imp_diff")
    print("Closer to zero --> more spurious dimension removed")
    for cidx,cat in enumerate(classifier.data_args["cat_list"]):
        initial_negativity = np.sum(cat_imp_diff_dict[cat])
        negativity_left = np.sum(gate_arr*cat_imp_diff_dict[cat])
        print("\n=========================================================")
        print("Cat:{}\tInitial:{:0.6f}\tNegLeft:{:0.6f}\tDelta:{:0.6f}".format(
                                            cat,
                                            initial_negativity,
                                            negativity_left,
                                            negativity_left-initial_negativity,
                )
            )
        #Printing the spurious dimension in this domain
        print("dimension spuriousness score:")
        dim_wise_diff = [
                (didx,cat_imp_diff_dict[cat][didx]) 
                        for didx in range(cat_imp_diff_dict[cat].shape[0])
        ]
        dim_wise_diff.sort(key=lambda x:x[-1])
        mypp(dim_wise_diff)
    
    #Printing the global differnet dimension wise
    dim_wise_diff = [
                (didx,global_imp_diff[didx])
                    for didx in range(global_imp_diff.shape[0])
    ]
    dim_wise_diff.sort(key=lambda x: x[-1])
    print("\n==============================================")
    print("Global imp difference: Spuriousness Score:")
    mypp(dim_wise_diff)

    return gate_arr



def get_dimension_gate(data_args,model_args):
    '''
    '''
    #If we dont want to do any gating we will pass a constant tensor with one
    if(model_args["gate_weight_exp"]==None):
        all_ones = tf.constant(
                        tf.ones([1,model_args["bemb_dim"]])
        )

        return all_ones

    print("====================================================")
    print("PREPARING THE GATE TO THE SHADOWS OF MORDOR")
    print("====================================================")
    #Loaidng the previous run from which we want to get dim variance
    indo_vacc,ood_vacc,classifier = evaluate_ood_indo_performance(
                                                    data_args=data_args,
                                                    model_args=model_args,
                                                    purpose="gate",
                                                    only_indo=False,
    )

    #Getting the importaance score for each of the dimension
    sent_weights = get_sent_imp_weights(classifier)

    #Getting the dim variance using the task importance
    #Getting the spuriousness of each dimension due to domain variation
    # dim_std = np.std(sent_weights,axis=1)

    #Using the performance drop idea for spuriousness clacluation
    gate_arr = get_feature_spuriousness(classifier,ood_vacc,sent_weights)

    #Now we will have a cutoff for the spurousness
    # cutoff_idx = int(dim_std.shape[0]*model_args["gate_var_cutoff"])
    # sorted_dim_idx = np.argsort(dim_std)[cutoff_idx]
    # cutoff_value = dim_std[sorted_dim_idx]
    # gate_arr = (dim_std<=cutoff_value)*1.0
    # gate_arr = np.expand_dims(gate_arr,axis=0)
    # print("cutoff_percent:{}\tcutoff_idx:{}\tcutoff_val:{}\tnum_alive:{}".format(
    #                             model_args["gate_var_cutoff"],
    #                             cutoff_idx,
    #                             cutoff_value,
    #                             np.sum(gate_arr)
    # ))

    #Creating the gate tensor
    gate_tensor = tf.constant(
                        gate_arr,
                        dtype=tf.float32
    )

    #Deleting the classifier for safely
    del classifier
    return gate_tensor

def evaluate_ood_indo_performance(data_args,model_args,purpose,only_indo=False):
    '''
    Given a trained classifier for the task we will try to retreive the
    OOD and INDO performance for all the classifier on the validation set

    Remember we are using the validation as lower end of the batch and since
    we dont shuffle the dataset suring training, the validation set are not
    seen during training.

    Output Contract:
    {
        domain 1 (classfier) : {
                                    domain 1 (data): vacc
                                    domain 2 (data): vacc
                                    ...
                                }
        ...
    }
    '''
    #First of all we will load the gate array we are truing to run gate experiemnt
    # gate_tensor = None #get_dimension_gate(data_args,model_args)

    #First of all creating the model
    classifier = TransformerClassifier(data_args,model_args)

    #Now we will compile the model
    classifier.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )

    #Creating the dataset object
    # data_handler = DataHandleTransformer(data_args)
    # all_cat_ds,all_topic_ds = data_handler.amazon_reviews_handler()
    all_cat_ds = data_args["all_cat_ds"]

    if purpose=="load":
        load_path = "{}/cp_cat_{}.ckpt".format(
                                    data_args["load_weight_path"],
                                    model_args["load_weight_epoch"]
        )
        load_dir = os.path.dirname(load_path)
        classifier.load_weights(load_path)

        #Loading the gate tensor
        gate_fpath = "{}/gate_tensor.npz".format(data_args["load_weight_path"])
        gate_arr = np.load(gate_fpath)["gate_tensor"]
        gate_tensor = tf.constant(
                        gate_arr,
                        dtype=tf.float32
        )

    elif purpose=="gate":
        checkpoint_path = "{}/cp_{}.ckpt".format(
                                    data_args["gate_weight_path"],
                                    model_args["gate_weight_epoch"]
        )
        checkpoint_dir = os.path.dirname(checkpoint_path)
        classifier.load_weights(checkpoint_path)

        #Here generally we will use the non-gated experiemnts to generally all will be ones
        print("Warning! All the gates are open")
        gate_arr = np.ones((1,model_args["bemb_dim"]))
        gate_tensor = tf.constant(
                        gate_arr,
                        dtype=tf.float32
        )
    else:
        raise NotImplementedError()
    

    #Iterating over all the classifier
    indo_vacc = {}
    ood_vacc = defaultdict(dict)
    for cidx,cname in enumerate(data_args["cat_list"]):
        #Iterating over all the dataset for both indo and OOD
        for cat,cat_ds in all_cat_ds.items():
            #If we just want indomain accuracy
            if(only_indo==True and cat!=cname):
                continue
            
            print("Getting ood perf of :{}\t on:{}".format(cname,cat))
            #Getting a new accuracy op for fear of updating old one
            acc_op = keras.metrics.Mean(name="temp_acc_op")
            acc_op.reset_states()

            #Going over all the batches of this catefory for given classifier
            for data_batch in cat_ds:
                classifier.valid_step_stage1(cidx,data_batch,gate_tensor,acc_op,"sentiment")
            
            vacc = acc_op.result().numpy()

            #Now assignign the accuracy to appropriate place
            if cname==cat:
                indo_vacc[cname]=float(vacc)
            ood_vacc[cname][cat]=float(vacc)
        
        print("Indomain VAcc: ",indo_vacc[cname])
        print("Outof Domain Vacc: ")
        mypp(ood_vacc[cname])
    
    #Also we need to get the indomain accuracy of the topics
    # topic_indo_vacc={}
    # for tidx,tname in enumerate(data_args["topic_list"]):
    #     print("Getting indo perf of :{}".format(tname))
    #     #Getting a new accuracy op for fear of updating old one
    #     acc_op = keras.metrics.Mean(name="temp_acc_op")
    #     acc_op.reset_states()

    #     #Going over all the batches of this catefory for given classifier
    #     for data_batch in all_topic_ds[tname]:
    #         classifier.valid_step(tidx,data_batch,gate_tensor,acc_op,"topic")
        
    #     vacc = acc_op.result().numpy()
    #     topic_indo_vacc[tname]=vacc 

    print("========================================")
    print("Indomain Validation Accuracy:") 
    mypp(indo_vacc)
    print("OutOfDomain Validation Accuracy:")
    mypp(ood_vacc)
    # print("OutOfDomain Topic Validation Accuracy:")
    # mypp(topic_indo_vacc)
    print("========================================")
    
    #Getting the overall OOD difference
    print("Getting the overall OOD drop!")
    overall_drop = 0.0
    all_drop_list = []
    all_delta_list=[]
    all_ood_vacc=[]
    all_indo_vacc = []
    for cat in classifier.data_args["cat_list"]:
        cat_drop = 0.0
        all_indo_vacc.append(ood_vacc[cat][cat])
        for dcat in classifier.data_args["cat_list"]:
            drop = ood_vacc[cat][dcat] - ood_vacc[cat][cat]
            #Getting all the domain delta
            if cat!=dcat:
                all_delta_list.append(drop)
            
            #Just adding the drop
            if drop<0:
                cat_drop+=drop
                all_drop_list.append(drop)
                
            #Addng the validation accuracy to the list
            all_ood_vacc.append(ood_vacc[cat][dcat])
        print("cat:{}\tdrop:{}".format(cat,cat_drop))
        overall_drop+=cat_drop
    print("Overall Drop:",overall_drop)
    print("Overall variance in the OOD performace:",np.std(all_delta_list))
    print("Mean Drop in OOD Vacc",np.mean(all_drop_list))
    print("Mean OOD Valid-Acc:",np.mean(all_ood_vacc))
    print("Mean INDO Valid-Acc:",np.mean(all_indo_vacc))
    print("Avg OOD vacc / Avg OOD vacc drop:",np.mean(all_ood_vacc)/abs(np.mean(all_drop_list)))

    #Saving these in the results
    ood_meta_dict = {}
    ood_meta_dict["overall_ood_drop"]       = float(np.sum(all_drop_list))
    ood_meta_dict["mean_ood_drop"]          = float(np.mean(all_drop_list))

    ood_meta_dict["overall_ood_std"]        = float(np.std(all_delta_list))

    ood_meta_dict["mean_all_vacc"]          = float(np.mean(all_ood_vacc))
    ood_meta_dict["mean_indo_vacc"]         = float(np.mean(all_indo_vacc))

    ood_meta_dict["avg_vacc_by_avg_drop"]   = float(np.mean(all_ood_vacc)/abs(np.mean(all_drop_list)))
    ood_meta_dict["avg_vacc_by_sum_drop"]   = float(np.mean(all_ood_vacc)/abs(np.sum(all_drop_list)))

    
    #Now we have all the indo and ood validation accuracy
    return indo_vacc,ood_vacc,classifier,ood_meta_dict

def get_spuriousness_rank(classifier,policy):
    #Now getting the importance weights of the topics
    topic_weights=[
        np.squeeze(tf.sigmoid(classifier.topic_importance_weight_list[tidx]).numpy())
                        for tidx in range(len(classifier.data_args["topic_list"]))
    ]
    new_topic_weights = []
    for tidx in range(len(classifier.data_args["topic_list"])):
        timp_weight = np.expand_dims(topic_weights[tidx],axis=0)
        tfirst_weight = np.abs(classifier.topic_classifier_list[tidx].get_weights()[0].T)
        
        new_timp_weight = np.sum(tfirst_weight*timp_weight,axis=0)
        new_topic_weights.append(new_timp_weight)
    
    topic_weights = new_topic_weights


    #Getting the spuriousness of each dimension due to domain variation
    sent_weights = get_sent_imp_weights(classifier)


    #Multiplying by validation accuracy to correct for presence of importance
    topic_validation_accuracy = [ 
        dict(
            tidx=tidx,
            vacc=classifier.topic_valid_acc_list[tidx].result().numpy()
        )
                for tidx in range(len(classifier.data_args["topic_list"]))
    ]
    print("Topic_validation Accuracy:")
    mypp(topic_validation_accuracy)


    #Getting the spurious ranking (decreasing order)
    dim_std = np.std(sent_weights,axis=1)
    dim_std_norm = np.linalg.norm(dim_std)
    dim_spurious_rank = np.argsort(-1*dim_std)

    #Now calculating the correlaiton
    topic_correlation = []
    for tidx,tname in enumerate(classifier.data_args["topic_list"]):
        #Getting the correlation
        topic_rank = np.argsort(-1*topic_weights[tidx])
        if policy=="weightedtau":
            tau,_=scipy.stats.weightedtau(dim_spurious_rank,topic_rank,rank=False)
            topic_correlation.append((tau,tname))
        elif policy=="kendalltau":
            tau,_=scipy.stats.weightedtau(dim_spurious_rank,topic_rank)
            topic_correlation.append((tau,tname))
        elif policy=="spearmanr":
            #correlation between the rank array (take abs)
            corr,_ = scipy.stats.spearmanr(dim_spurious_rank,topic_rank)
            topic_correlation.append((abs(corr),tname))
        elif "dot" in policy:
            #Getting dot product between the spuriousness vector and topic importance vector
            topic_dot = np.sum(dim_std*topic_weights[tidx])

            #Multiplying by validation accuracy to correct for presence of importance
            if "weighted" in policy:
                topic_vacc = topic_validation_accuracy[tidx]["vacc"]
                topic_dot = topic_dot*topic_vacc

            topic_correlation.append((topic_dot,tname))
        elif "cosine" in policy:
            topic_norm = np.linalg.norm(topic_weights[tidx])
            topic_cos = np.sum(dim_std*topic_weights[tidx])/(topic_norm*dim_std_norm)

            #Multiplying by validation accuracy to correct for presence of importance
            if "weighted" in policy:
                topic_vacc = topic_validation_accuracy[tidx]["vacc"]
                topic_cos = topic_cos*topic_vacc
            
            topic_correlation.append((topic_cos,tname))
        elif policy=="weighted_dot":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
    #Sorting the ranking
    topic_correlation.sort(key=lambda x:-x[0])
    print("\ncorrelation_policy: ",policy)
    mypp(topic_correlation)

    topic_correlation_dict = {}
    for corr_val,tname in topic_correlation:
        topic_correlation_dict[tname]=corr_val

    return topic_correlation,topic_correlation_dict

def load_and_analyze_transformer(data_args,model_args):
    # #First of all creating the model
    # classifier = TransformerClassifier(data_args,model_args)
    
    
    # checkpoint_path = "nlp_logs/{}/cp_{}.ckpt".format(model_args["expt_name"],model_args["load_weight_epoch"])
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    
    
    # classifier.load_weights(checkpoint_path)

    #Getting the classifier with the loaded weights and getting the ood accuracy
    indo_vacc,ood_vacc,classifier,ood_meta_dict = evaluate_ood_indo_performance(
                                            data_args=data_args,
                                            model_args=model_args,
                                            only_indo=False,
                                            purpose="load",
    )
    return indo_vacc,ood_vacc,ood_meta_dict

    #Getting the spuriousness score using the drop in importance idea
    # sent_weights = get_sent_imp_weights(classifier)
    # gate_arr = get_feature_spuriousness(classifier,ood_vacc,sent_weights)


    # #Printing the different correlation weights
    # get_spuriousness_rank(classifier,"weightedtau")
    # get_spuriousness_rank(classifier,"kendalltau")
    # get_spuriousness_rank(classifier,"spearmanr")
    # _,corr_dict_dot     = get_spuriousness_rank(classifier,"dot")
    # _,corr_dict_wdot    = get_spuriousness_rank(classifier,"weighted_dot")
    # _,corr_dict_cos     = get_spuriousness_rank(classifier,"cosine")
    # _,corr_dict_wcos    = get_spuriousness_rank(classifier,"weighted_cosine")

    # return corr_dict_dot,corr_dict_cos

    # dim_score = np.mean(sent_weights,axis=1) * np.std(sent_weights,axis=1)
    # spurious_dims = np.argsort(dim_score)

    # dim_std = np.std(sent_weights,axis=1)
    # dim_mean=np.mean(sent_weights,axis=1)
    # dim_score2 = dim_std * (dim_mean>0.7)
    # num_spurious = np.sum(dim_score2>0.1)
    # spurious_dims2 = np.argsort(dim_score2)[-num_spurious:]


    # #Sorting the importance score of the topic weights
    # topic_weight = np.squeeze(tf.sigmoid(classifier.topic_importance_weight_list[0]).numpy())
    # topic_imp_dims = np.argsort(topic_weight)

    # #Percentage of spuriousness is biggest 50 importance
    # upto_index = num_spurious
    # spuriousness_percentage = len(set(spurious_dims2).intersection(set(topic_imp_dims[-1*upto_index:])))/upto_index
    # pdb.set_trace()
    return 

def dump_arguments(arg_dict,expt_name,fname):
    #This function will dump the arguments in a file for tracking purpose
    filepath = "{}/{}.txt".format(expt_name,fname)
    with open(filepath,"w") as whandle:
        json.dump(arg_dict,whandle,indent="\t")


################################################################
#########  Stage 1: Parallel Job Runner for all Subset #########
################################################################
def run_parallel_jobs_subset_exp(data_args,model_args):
    '''
    This job runner will run the subset of all the topic dimension in the 
    to see if we are indeed able to find the correct causal subset 
    using our definition 1 of supuriousness.

    Definition: The causal subset will have minimum total drop in performance
    '''
    #First of all we need to get all the feature subset we want to make
    all_subset_loc=[]
    index_list = range(0,len(data_args["topic_list"]))
    for deg in range(0,len(data_args["topic_list"])+1):
        #Getting the subset location
        all_subset_loc += list(combinations(index_list,deg))
    
    #Creating one single dataset for time saving and uniformity
    data_handler = DataHandleTransformer(data_args)
    if "amazon" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.amazon_reviews_handler()
    elif "nlp_toy" in data_args["path"]:
        all_cat_ds,all_topic_ds,new_all_cat_df = data_handler.toy_nlp_dataset_handler()
    else:
        raise NotImplementedError()

    #Creating all the experiment metadata
    all_expt_config = []
    for sub in all_subset_loc:
        alive_dims = tuple(set(index_list).difference(sub))
        config = {}
        config["mask_feature_dims"]=sub
        config["alive_feature_dims"] = alive_dims
        config["data_args"]=data_args.copy()
        config["model_args"]=model_args.copy()

        #Now changing the mask argument in the data args
        config["data_args"]["mask_feature_dims"]=sub
        config["data_args"]["expt_name"]="exp.synct.{}".format(alive_dims)
        config["model_args"]["expt_name"]="exp.synct.{}".format(alive_dims)

        
        #Now we have the cat_df to the config for reuse of dataset creation
        config["data_args"]["new_all_cat_df"]=new_all_cat_df.copy()

        #Adding the experiment config
        if len(alive_dims)==1:
            all_expt_config.append(config)
    

    #Now we will start the parallel experiment
    ncpus = 6#int(3.0/4.0*multiprocessing.cpu_count())
    with Pool(ncpus) as p:
            production = p.map(worker_kernel,all_expt_config)
    
    return 

#Creating the worker kernel
def worker_kernel(problem_config):
    #First of we have to create the masked dataset
    data_handler = DataHandleTransformer(problem_config["data_args"])
    #Creating the zipped dataset
    print("Creating the masked dataset for worker:{}".format(problem_config["alive_feature_dims"]))
    all_cat_ds = {}
    for cat in problem_config["data_args"]["cat_list"]:
        cat_df = problem_config["data_args"]["new_all_cat_df"][cat]
        #Getting the dataset object
        cat_ds = data_handler._convert_df_to_dataset_stage1(
                                    df=cat_df,
                                    doc_col_name="topic_feature",
                                    label_col_name="label",
                                    mask_feature_dims=problem_config["mask_feature_dims"],
                                    # topic_col_name="topic",
                                    # topic_weight_col_name="topic_weight",
        )
        all_cat_ds[cat]=cat_ds
    problem_config["data_args"]["all_cat_ds"]=all_cat_ds

    print("Training the worker:{}".format(problem_config["alive_feature_dims"]))
    expt_main_path = problem_config["data_args"]["expt_meta_path"]
    #Getting the arguments
    data_args = problem_config["data_args"]
    model_args = problem_config["model_args"]

    #Creating the metadata folder
    data_args["expt_meta_path"] = "{}/{}".format(
                                            data_args["expt_meta_path"],
                                            model_args["expt_name"]
    )
    os.makedirs(data_args["expt_meta_path"],exist_ok=True)

    #Starting the training phase
    transformer_trainer_stage1(data_args,model_args)

    #Now starting the validation run
    print("Validating the worker:{}".format(problem_config["alive_feature_dims"]))
    data_args["load_weight_path"]=data_args["expt_meta_path"]
    model_args["load_weight_epoch"]=model_args["epochs"]-1
    indo_vacc,ood_vacc,ood_meta_dict = load_and_analyze_transformer(data_args,model_args)

    #Adding the result to the config
    problem_config["ood_meta_dict"]=ood_meta_dict
    problem_config["ood_vacc"]=ood_vacc
    problem_config["indo_vacc"]=indo_vacc


    #Removing the dataset related objects
    del problem_config["data_args"]["new_all_cat_df"]
    del problem_config["data_args"]["all_cat_ds"]
    #Dumping the result in one json
    import json
    fname = "{}/results_{}.json".format(
                            expt_main_path,
                            problem_config["alive_feature_dims"]
    )
    with open(fname,"w") as fp:
        json.dump(problem_config,fp,indent="\t")
    
    return problem_config


def set_gpu(self,gpu_num):
    '''
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            print("Using GPU No: {}".format(gpu_num))
            tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
       

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-expt_num',dest="expt_name",type=str)
    parser.add_argument('-num_samples',dest="num_samples",type=int,default=None)
    parser.add_argument('-num_topics',dest="num_topics",type=int)
    parser.add_argument('-num_topic_samples',dest="num_topic_samples",type=int,default=None)
    parser.add_argument('-l1_lambda',dest="l1_lambda",type=float,default=0.0)
    parser.add_argument('-lr',dest="lr",type=float)

    parser.add_argument('-path',dest="path",type=str)
    parser.add_argument('-task_name',dest="task_name",type=str,default="sentiment")
    parser.add_argument('-spurious_ratio',dest="spurious_ratio",type=float,default=None)
    parser.add_argument('-causal_ratio',dest="causal_ratio",type=float,default=None)

    #Arguments related to the main convergence experiemtn and subsequent INLP or AdvRem
    parser.add_argument('-topic0_corr',dest="topic0_corr",type=float,default=None)
    parser.add_argument('-topic1_corr',dest="topic1_corr",type=float,default=None)
    parser.add_argument('-noise_ratio',dest="noise_ratio",type=float,default=None)
    parser.add_argument('-main_topic',dest="main_topic",type=int,default=None)
    parser.add_argument('-num_hidden_layer',dest="num_hidden_layer",type=int,default=None)
    parser.add_argument('-removal_mode',dest="removal_mode",type=str,default=None)
    parser.add_argument('-main_model_mode',dest="main_model_mode",type=str,default=None)
    parser.add_argument('-head_retrain_mode',dest="head_retrain_mode",type=str,default=None)
    parser.add_argument('-l2_lambd',dest="l2_lambd",type=float,default=None)
    parser.add_argument('-loss_type',dest="loss_type",type=str,default=None)
    parser.add_argument('-dtype',dest="dtype",type=str,default=None)
    parser.add_argument('-inv_dims',dest="inv_dims",type=int,default=None)
    parser.add_argument('-tab_sigma_ubound',dest="tab_sigma_ubound",type=float,default=None)
    parser.add_argument('-dropout_rate',dest="dropout_rate",type=float,default=None)
    parser.add_argument('--save_emb',default=False,action="store_true")
    parser.add_argument('-neg_topic_corr',dest="neg_topic_corr",type=float,default=None)
    parser.add_argument('-neg1_flip_method',dest="neg1_flip_method",type=str,default=None)
    parser.add_argument('--measure_flip_pdelta',default=False,action="store_true")
    parser.add_argument('-gpu_num',dest="gpu_num",type=int,default=0)

    

    #Arguments related to the adversarial removal
    parser.add_argument('-adv_rm_epochs',dest="adv_rm_epochs",type=int,default=None)
    parser.add_argument('-rev_grad_strength',dest="rev_grad_strength",type=float,default=None)
    parser.add_argument('-adv_rm_method',dest="adv_rm_method",type=str,default=None)

    parser.add_argument('-stage',dest="stage",type=int)
    #parser.add_argument('-bemb_dim',dest="bemb_dims",type=int)
    parser.add_argument('-debug_cidx',dest="debug_cidx",type=int,default=None)
    parser.add_argument('-debug_tidx',dest="debug_tidx",type=int,default=None)
    parser.add_argument('--reverse_grad',default=False,action="store_true")

    parser.add_argument('-topic_epochs',dest="topic_epochs",type=int,default=None)
    parser.add_argument('-num_proj_iter',dest="num_proj_iter",type=int,default=None)
    parser.add_argument('-hlayer_dim',dest="hlayer_dim",type=int,default=None)
    

    parser.add_argument('-emb_path',dest="emb_path",type=str,default=None)

    parser.add_argument('--extend_topic_set',default=False,action="store_true")
    parser.add_argument('-vocab_path',dest="vocab_path",type=str,default="assets/word2vec_10000_200d_labels.tsv")
    parser.add_argument('-num_neigh',dest="num_neigh",type=int,default=None)

    parser.add_argument('--train_emb',default=False,action="store_true")
    parser.add_argument('--normalize_emb',default=False,action="store_true")

    parser.add_argument("-temb_dim",dest="temb_dim",type=int,default=None)
    parser.add_argument("--normalize_temb",default=False,action="store_true")

    parser.add_argument('-tfreq_ulim',dest="tfreq_ulim",type=float,default=1.0)
    parser.add_argument('-transformer',dest="transformer",type=str,default="bert-base-uncased")
    parser.add_argument('-num_epochs',dest="num_epochs",type=int)
    
    parser.add_argument("-load_weight_exp",dest="load_weight_exp",type=str,default=None)
    parser.add_argument("-load_weight_epoch",dest="load_weight_epoch",type=str,default=None)
    
    parser.add_argument("-gate_weight_exp",dest="gate_weight_exp",type=str,default=None)
    parser.add_argument("-gate_weight_epoch",dest="gate_weight_epoch",type=int,default=None)
    parser.add_argument("-gate_var_cutoff",dest="gate_var_cutoff",type=str)

    parser.add_argument('--bert_as_encoder',default=False,action="store_true")
    parser.add_argument('--train_bert',default=False,action="store_true")
    parser.add_argument('--cached_bemb',default=False,action="store_true")

    args=parser.parse_args()
    print(args)
    #Setting the GPU
    set_gpu(args.gpu_num)

    #Defining the Data args
    data_args={}
    data_args["path"] = args.path                   #"dataset/amazon/"
    data_args["task_name"] = args.task_name         #"sentiment or regard ..."
    data_args["causal_ratio"] = args.causal_ratio
    data_args["spurious_ratio"] = args.spurious_ratio
    data_args["stage"]=args.stage                    #Whether we are in stage1 or stage2
    data_args["debug_cidx"]=args.debug_cidx
    data_args["debug_tidx"]=args.debug_tidx
    data_args["transformer_name"]=args.transformer
    data_args["num_class"]=2
    data_args["max_len"]=200
    data_args["num_sample"]=args.num_samples
    data_args["num_topic_samples"]=args.num_topic_samples
    if args.transformer=="bert-base-uncased":
        data_args["batch_size"]=32
    elif args.transformer=="bert-large-uncased":
        data_args["batch_size"]=8
    else:
        data_args["batch_size"]=32
    
    data_args["shuffle_size"]=data_args["batch_size"]*3
    if "amazon" in data_args["path"]:
        data_args["cat_list"]=["arts","books","phones","clothes","groceries","movies","pets","tools"]
        data_args["topic_corr_list"]=[args.topic0_corr,args.topic1_corr]
    elif "nlp_toy2" in data_args["path"]:
        data_args["topic_corr_list"]=[args.topic0_corr,args.topic1_corr]
        data_args["noise_ratio"]=args.noise_ratio
        data_args["main_topic"]=args.main_topic
        data_args["dtype"]=args.dtype
        data_args["inv_dims"]=args.inv_dims
        data_args["sp_dims"]=args.inv_dims
        data_args["tab_sigma_ubound"]=args.tab_sigma_ubound
    elif "nlp_toy" in data_args["path"]:
        data_args["cat_list"]=["gender","race","orientation"]
    elif "multinli" in data_args["path"]:
        data_args["neg_topic_corr"]=args.neg_topic_corr
        data_args["noise_ratio"]=args.noise_ratio
        data_args["dtype"]=args.dtype

    data_args["num_topics"]=args.num_topics
    data_args["topic_list"]=list(range(data_args["num_topics"]))
    data_args["per_topic_class"]=2 #Each of the topic is binary (later could have more)
    data_args["tfreq_ulim"]=args.tfreq_ulim
    data_args["lda_epochs"]=25
    data_args["min_df"]=0.0
    data_args["max_df"]=1.0
    data_args["emb_path"]=args.emb_path
    data_args["extend_topic_set"]=args.extend_topic_set
    data_args["vocab_path"]=args.vocab_path
    data_args["num_neigh"]=args.num_neigh
    # data_args["mask_feature_dims"]=list(range(4,len(data_args["topic_list"])))
    data_args["save_emb"]=args.save_emb
    data_args["neg1_flip_method"]=args.neg1_flip_method

    #Defining the Model args
    model_args={}
    model_args["expt_name"]=args.expt_name
    data_args["expt_name"]=model_args["expt_name"]
    model_args["load_weight_exp"]=args.load_weight_exp
    model_args["load_weight_epoch"]=args.load_weight_epoch
    model_args["lr"]=args.lr
    model_args["epochs"]=args.num_epochs
    model_args["l1_lambda"]=args.l1_lambda
    model_args["valid_split"]=0.2
    model_args["train_bert"]=args.train_bert
    model_args["cached_bemb"]=args.cached_bemb
    if args.bert_as_encoder==True:
        if args.transformer=="bert-base-uncased":
            model_args["bemb_dim"] = 768
        elif args.transformer=="bert-large-uncased":
            model_args["bemb_dim"] = 1024
    else:
        model_args["bemb_dim"] = 768 if args.stage==2 else len(data_args["topic_list"]) #The dimension of bert produced last layer
    model_args["hlayer_dim"]=args.hlayer_dim
    model_args["temb_dim"] = args.temb_dim
    model_args["normalize_temb"] = args.normalize_temb
    model_args["shuffle_topic_batch"]=False
    model_args["gate_weight_exp"]=args.gate_weight_exp
    model_args["gate_weight_epoch"]=args.gate_weight_epoch
    model_args["gate_var_cutoff"]=args.gate_var_cutoff
    model_args["reverse_grad"]=args.reverse_grad
    model_args["rev_grad_strength"]=args.rev_grad_strength
    model_args["train_emb"]=args.train_emb
    model_args["bert_as_encoder"]=args.bert_as_encoder
    model_args["normalize_emb"]=args.normalize_emb
    model_args["num_hidden_layer"] = args.num_hidden_layer
    model_args["num_proj_iter"]=args.num_proj_iter
    model_args["topic_epochs"]=args.topic_epochs
    model_args["adv_rm_epochs"]=args.adv_rm_epochs
    model_args["removal_mode"]=args.removal_mode
    model_args["adv_rm_method"]=args.adv_rm_method
    model_args["main_model_mode"]=args.main_model_mode
    model_args["head_retrain_mode"] = args.head_retrain_mode
    model_args["l2_lambd"]=args.l2_lambd
    model_args["loss_type"]=args.loss_type
    model_args["dropout_rate"]=args.dropout_rate
    model_args["measure_flip_pdelta"]=args.measure_flip_pdelta
    model_args["gpu_num"]=args.gpu_num

    #Creating the metadata folder
    meta_folder = "nlp_logs/{}".format(model_args["expt_name"])
    os.makedirs(meta_folder,exist_ok=True)
    data_args["expt_meta_path"]=meta_folder

    #################################################
    #        SELECT ONE OF THE JOBS FROM BELOW      #
    #################################################
    # transformer_trainer_stage2(data_args,model_args)
    # transformer_trainer_stage2_inlp(data_args,model_args)
    nbow_trainer_stage2(data_args,model_args)
    # run_parallel_jobs_subset_exp(data_args,model_args)
    # transformer_trainer(data_args,model_args)
    # load_and_analyze_transformer(data_args,model_args)


            


