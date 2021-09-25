import numpy as np
import pandas as pd
import scipy


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


from nlp_data_handle import *

class TransformerClassifier(keras.Model):
    '''
    This class will have the main model which will pass through the
    transformer and train 
    '''
    def __init__(self,data_args,model_args):
        super(TransformerClassifier,self).__init__()
        self.data_args = data_args
        self.model_args = model_args

        #Now initializing the layers to be used
        if "distil" in data_args["transformer_name"]:
            self.bert_model = TFDistilBertModel.from_pretrained(data_args["transformer_name"])
        else:
            self.bert_model = TFBertModel.from_pretrained(data_args["transformer_name"])
        if model_args["train_bert"]==False:
            for layer in self.bert_model.layers:
                layer.trainable = False

        #Initializing the heads for each classifier (Domain Sntiment)
        self.cat_classifier_list = []
        self.cat_importance_weight_list = []
        self.cat_temb_importance_weight_list = []
        for cat in self.data_args["cat_list"]:
            #Creating the imporatance paramaters for each classifier
            cat_imp_weight = tf.Variable(
                tf.random_normal_initializer(mean=0.0,stddev=1.0)(
                                shape=[1,model_args["bemb_dim"]],
                                dtype=tf.float32,
                ),
                trainable=True
            )
            self.cat_importance_weight_list.append(cat_imp_weight)

            #Creating the topic embedding weights to be used by classifier
            cat_temb_imp_weight = tf.Variable(
                tf.random_normal_initializer(mean=0.0,stddev=1.0)(
                                shape=[1,1,len(data_args["topic_list"])]
                ),
                trainable=True
            )
            self.cat_temb_importance_weight_list.append(cat_temb_imp_weight)

            #Creating a dense layer
            cat_dense = layers.Dense(2,activation="softmax")
            self.cat_classifier_list.append(cat_dense)
        
        #Initializing the heads for the "interpretable" topics
        self.topic_classifier_list = []
        self.topic_embedding_layer_list = []
        self.topic_importance_weight_list = []
        for topic in self.data_args["topic_list"]:
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
            topic_embed_layer = layers.Dense(self.model_args["temb_dim"])
            self.topic_embedding_layer_list.append(topic_embed_layer)

            #Creating a dense layer
            topic_dense = layers.Dense(self.data_args["per_topic_class"],activation="softmax")
            self.topic_classifier_list.append(topic_dense)



        #Initializing the trackers for sentiment classifier
        self.sent_pred_xentropy = keras.metrics.Mean(name="sent_pred_x")
        self.sent_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="c{}_valid_acc".format(cat))
                for cat in self.data_args["cat_list"]
        ]
        self.sent_l1_loss_list = [ 
            tf.keras.metrics.Mean(name="c{}_l1_loss".format(cat))
                for cat in self.data_args["cat_list"]
        ]

        #Initilaizing the trackers for topics classifier
        self.topic_pred_xentropy = keras.metrics.Mean(name="topic_pred_x")
        self.topic_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="t{}_valid_acc".format(topic))
                for topic in self.data_args["topic_list"]
        ]
        self.topic_l1_loss_list = [ 
            tf.keras.metrics.Mean(name="t{}_l1_loss".format(cat))
                for cat in self.data_args["topic_list"]
        ]
    
    def reset_all_metrics(self):
        self.sent_pred_xentropy.reset_state()
        self.topic_pred_xentropy.reset_state()
        for tidx in range(len(self.data_args["topic_list"])):
            self.topic_valid_acc_list[tidx].reset_state()
        for cidx in range(len(self.data_args["cat_list"])):
            self.sent_valid_acc_list[cidx].reset_state()

    def compile(self, optimizer):
        super(TransformerClassifier, self).compile()
        self.optimizer = optimizer 
    
    def get_sentiment_pred_prob(self,idx_train,mask_train,gate_tensor,cidx):     
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

        #Now we will gate the dimension which are spurious
        gated_avg_embedding = avg_embedding*gate_tensor
        
        #Now we will apply the dense layer for this category
        cat_class_prob = self.cat_classifier_list[cidx](gated_avg_embedding)

        return cat_class_prob
    
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
    
    def train_step(self,sidx,single_ds,gate_tensor,task):
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

        #Taking aside a chunk of data for validation
        valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

        #Getting the train data
        label_train = label[0:valid_idx]
        idx_train = idx[0:valid_idx]
        mask_train = mask[0:valid_idx]

        #Getting the validation data
        label_valid = label[valid_idx:]
        idx_valid = idx[valid_idx:]
        mask_valid = mask[valid_idx:]

        if task=="sentiment":
            #First of all we need to get the topic embedding from all the 
            with tf.GradientTape() as tape:
                #Forward propagating the model
                train_prob = self.get_sentiment_pred_prob_topic_basis(idx_train,mask_train,gate_tensor,sidx)
                
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
            valid_prob = self.get_sentiment_pred_prob_topic_basis(idx_valid,mask_valid,gate_tensor,sidx)
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

    def valid_step(self,sidx,single_ds,gate_tensor,acc_op,task):
        '''
        Given a single dataset we will get the validation score
        (using the lower end of the dataset which was not used for training)
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
        if task=="sentiment":
            valid_prob = self.get_sentiment_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
        elif task=="topic":
            valid_prob = self.get_topic_pred_prob(idx_valid,mask_valid,gate_tensor,sidx)
        else:
            raise NotImplementedError()

        #Getting the validation accuracy
        acc = tf.keras.metrics.sparse_categorical_accuracy(label_valid,valid_prob)
        acc_op.update_state(acc)


def transformer_trainer(data_args,model_args):
    '''
    '''
    #First of all we will load the gate array we are truing to run gate experiemnt
    gate_tensor = get_dimension_gate(data_args,model_args)
    #Saving the gate tensor 
    gate_fpath = "nlp_logs/{}/gate_tensor".format(model_args["expt_name"])
    np.savez(gate_fpath,gate_tensor=gate_tensor.numpy())

    #Dumping the model arguments
    dump_arguments(data_args,data_args["expt_name"],"data_args")
    dump_arguments(model_args,model_args["expt_name"],"model_args")

    #First of all creating the model
    classifier = TransformerClassifier(data_args,model_args)

    #Now we will compile the model
    classifier.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )

    #Creating the dataset object
    data_handler = DataHandleTransformer(data_args)
    all_cat_ds,all_topic_ds = data_handler.amazon_reviews_handler()

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
    for eidx in range(model_args["epochs"]):
        #Now first we will reset all the metrics
        classifier.reset_all_metrics()
        print("\n==============================================")
        print("Starting Epoch: ",eidx)
        print("==============================================")

        #Now its time to train the topics
        for tidx,(tname,topic_ds) in enumerate(all_topic_ds.items()):
            #Training the topic through all the batches
            for data_batch in topic_ds:
                classifier.train_step(tidx,data_batch,gate_tensor,"topic")
            
            #Pringitn ghte metrics
            print("topic:{}\tceloss:{:0.5f}\tl1_loss:{:0.5f}\tvacc:{:0.5f}".format(
                                            tname,
                                            classifier.topic_pred_xentropy.result(),
                                            classifier.topic_l1_loss_list[tidx].result(),
                                            classifier.topic_valid_acc_list[tidx].result(),
                    )
            )

        #Saving the paramenters
        checkpoint_path = "nlp_logs/{}/cp_topic_{}.ckpt".format(model_args["expt_name"],eidx)
        classifier.save_weights(checkpoint_path)
    

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
                classifier.train_step(cidx,data_batch,gate_tensor,"sentiment")

            #Now we will print the metric for this category
            print("cat:{}\tceloss:{:0.5f}\tl1_loss:{:0.5f}\tvacc:{:0.5f}".format(
                                            cat,
                                            classifier.sent_pred_xentropy.result(),
                                            classifier.sent_l1_loss_list[cidx].result(),
                                            classifier.sent_valid_acc_list[cidx].result(),
                    )
            )

        #Saving the paramenters
        checkpoint_path = "nlp_logs/{}/cp_cat_{}.ckpt".format(model_args["expt_name"],eidx)
        classifier.save_weights(checkpoint_path)
    
    #Printing the variance of the importance weight
    get_cat_temb_importance_weight_variance(classifier)


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
    data_handler = DataHandleTransformer(data_args)
    all_cat_ds,all_topic_ds = data_handler.amazon_reviews_handler()

    if purpose=="load":
        load_path = "nlp_logs/{}/cp_{}.ckpt".format(
                                    model_args["load_weight_exp"],
                                    model_args["load_weight_epoch"]
        )
        load_dir = os.path.dirname(load_path)
        classifier.load_weights(load_path)

        #Loading the gate tensor
        gate_fpath = "nlp_logs/{}/gate_tensor.npz".format(model_args["load_weight_exp"])
        gate_arr = np.load(gate_fpath)["gate_tensor"]
        gate_tensor = tf.constant(
                        gate_arr,
                        dtype=tf.float32
        )

    elif purpose=="gate":
        checkpoint_path = "nlp_logs/{}/cp_{}.ckpt".format(
                                    model_args["gate_weight_exp"],
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
            acc_op.reset_state()

            #Going over all the batches of this catefory for given classifier
            for data_batch in cat_ds:
                classifier.valid_step(cidx,data_batch,gate_tensor,acc_op,"sentiment")
            
            vacc = acc_op.result().numpy()

            #Now assignign the accuracy to appropriate place
            if cname==cat:
                indo_vacc[cname]=vacc
            ood_vacc[cname][cat]=vacc
        
        print("Indomain VAcc: ",indo_vacc[cname])
        print("Outof Domain Vacc: ")
        mypp(ood_vacc[cname])
    
    #Also we need to get the indomain accuracy of the topics
    topic_indo_vacc={}
    for tidx,tname in enumerate(data_args["topic_list"]):
        print("Getting indo perf of :{}".format(tname))
        #Getting a new accuracy op for fear of updating old one
        acc_op = keras.metrics.Mean(name="temp_acc_op")
        acc_op.reset_state()

        #Going over all the batches of this catefory for given classifier
        for data_batch in all_topic_ds[tname]:
            classifier.valid_step(tidx,data_batch,gate_tensor,acc_op,"topic")
        
        vacc = acc_op.result().numpy()
        topic_indo_vacc[tname]=vacc 

    print("========================================")
    print("Indomain Validation Accuracy:") 
    mypp(indo_vacc)
    print("OutOfDomain Validation Accuracy:")
    mypp(ood_vacc)
    print("OutOfDomain Topic Validation Accuracy:")
    mypp(topic_indo_vacc)
    print("========================================")
    
    #Now we have all the indo and ood validation accuracy
    return indo_vacc,ood_vacc,classifier

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
    indo_vacc,ood_vacc,classifier = evaluate_ood_indo_performance(
                                            data_args=data_args,
                                            model_args=model_args,
                                            only_indo=False,
                                            purpose="load",
    )    

    #Printing the different correlation weights
    get_spuriousness_rank(classifier,"weightedtau")
    get_spuriousness_rank(classifier,"kendalltau")
    get_spuriousness_rank(classifier,"spearmanr")
    _,corr_dict_dot     = get_spuriousness_rank(classifier,"dot")
    _,corr_dict_wdot    = get_spuriousness_rank(classifier,"weighted_dot")
    _,corr_dict_cos     = get_spuriousness_rank(classifier,"cosine")
    _,corr_dict_wcos    = get_spuriousness_rank(classifier,"weighted_cosine")

    return corr_dict_dot,corr_dict_cos

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

def dump_arguments(arg_dict,expt_name,fname):
    #This function will dump the arguments in a file for tracking purpose
    filepath = "nlp_logs/{}/{}.txt".format(expt_name,fname)
    with open(filepath,"w") as whandle:
        json.dump(arg_dict,whandle,indent="\t")
    

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-expt_num',dest="expt_name",type=str)
    parser.add_argument('-num_samples',dest="num_samples",type=int)
    parser.add_argument('-num_topics',dest="num_topics",type=int)
    parser.add_argument('-num_topic_samples',dest="num_topic_samples",type=int)
    parser.add_argument('-l1_lambda',dest="l1_lambda",type=float)

    parser.add_argument("-temb_dim",dest="temb_dim",type=int)
    parser.add_argument("--normalize_temb",default=False,action="store_true")

    parser.add_argument('-tfreq_ulim',dest="tfreq_ulim",type=float,default=1.0)
    parser.add_argument('-transformer',dest="transformer",type=str,default="bert-base-uncased")
    parser.add_argument('-num_epochs',dest="num_epochs",type=int)
    
    parser.add_argument("-load_weight_exp",dest="load_weight_exp",type=str,default=None)
    parser.add_argument("-load_weight_epoch",dest="load_weight_epoch",type=str,default=None)
    
    parser.add_argument("-gate_weight_exp",dest="gate_weight_exp",type=str,default=None)
    parser.add_argument("-gate_weight_epoch",dest="gate_weight_epoch",type=int,default=None)
    parser.add_argument("-gate_var_cutoff",dest="gate_var_cutoff",type=str)

    parser.add_argument('--train_bert',default=False,action="store_true")

    args=parser.parse_args()
    print(args)

    #Defining the Data args
    data_args={}
    data_args["path"] = "dataset/amazon/"
    data_args["transformer_name"]=args.transformer
    data_args["num_class"]=2
    data_args["max_len"]=200
    data_args["num_sample"]=args.num_samples
    data_args["num_topic_samples"]=args.num_topic_samples
    data_args["batch_size"]=32
    data_args["shuffle_size"]=data_args["batch_size"]*3
    data_args["cat_list"]=["arts","books","phones","clothes","groceries","movies","pets","tools"]
    data_args["num_topics"]=args.num_topics
    data_args["topic_list"]=list(range(data_args["num_topics"]))
    data_args["per_topic_class"]=2 #Each of the topic is binary (later could have more)
    data_args["tfreq_ulim"]=args.tfreq_ulim
    data_args["lda_epochs"]=25
    data_args["min_df"]=0.0
    data_args["max_df"]=1.0

    #Defining the Model args
    model_args={}
    model_args["expt_name"]=args.expt_name
    data_args["expt_name"]=model_args["expt_name"]
    model_args["load_weight_exp"]=args.load_weight_exp
    model_args["load_weight_epoch"]=args.load_weight_epoch
    model_args["lr"]=0.001
    model_args["epochs"]=args.num_epochs
    model_args["l1_lambda"]=args.l1_lambda
    model_args["valid_split"]=0.2
    model_args["train_bert"]=args.train_bert
    model_args["bemb_dim"] = 768        #The dimension of bert produced last layer
    model_args["temb_dim"] = args.temb_dim
    model_args["normalize_temb"] = args.normalize_temb
    model_args["shuffle_topic_batch"]=False
    model_args["gate_weight_exp"]=args.gate_weight_exp
    model_args["gate_weight_epoch"]=args.gate_weight_epoch
    model_args["gate_var_cutoff"]=args.gate_var_cutoff

    #Creating the metadata folder
    meta_folder = "nlp_logs/{}".format(model_args["expt_name"])
    os.makedirs(meta_folder,exist_ok=True)

    transformer_trainer(data_args,model_args)
    # load_and_analyze_transformer(data_args,model_args)


            


