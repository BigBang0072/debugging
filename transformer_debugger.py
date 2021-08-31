import numpy as np
from numpy.lib.function_base import append
import pandas as pd


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import TFBertModel,TFDistilBertModel

import pdb
import os
import sys
from pprint import pprint


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
        self.bert_model = TFBertModel.from_pretrained(data_args["transformer_name"])
        if model_args["train_bert"]==False:
            for layer in self.bert_model.layers:
                layer.trainable = False

        #Initializing the heads for each classifier (Domain Sntiment)
        self.cat_classifier_list = []
        self.cat_importance_weight_list = []
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

            #Creating a dense layer
            cat_dense = layers.Dense(2,activation="softmax")
            self.cat_classifier_list.append(cat_dense)
        
        #Initializing the heads for the "interpretable" topics
        self.topic_classifier_list = []
        self.topic_importance_weight_list = []
        # for topic in self.data_args["topic_list"]:
        #     #Creating the imporatance paramaters for each classifier
        #     topic_imp_weight = tf.Variable(
        #         tf.random_normal_initializer(mean=0.0,stddev=1.0)(
        #                         shape=[1,model_args["bemb_dim"]],
        #                         dtype=tf.float32,
        #         ),
        #         trainable=True
        #     )
        #     self.topic_importance_weight_list.append(topic_imp_weight)

        #     #Creating a dense layer
        #     topic_dense = layers.Dense(2,activation="softmax")
        #     self.topic_classifier_list.append(topic_dense)

        #Right now we will use just one classifier (we know domain should not be stable topics)
        self.topic_importance_weight_list.append(
            tf.Variable(
                tf.random_normal_initializer(mean=0.0,stddev=1.0)(
                                shape=[1,model_args["bemb_dim"]],
                                dtype=tf.float32,
                ),
                trainable=True
            )
        )
        self.topic_classifier_list.append(
            layers.Dense(len(self.data_args["topic_list"]),activation="softmax")
        )



        #Initializing the trackers for sentiment classifier
        self.sent_pred_xentropy = keras.metrics.Mean(name="sent_pred_x")
        self.sent_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="{}_valid_acc".format(cat))
                for cat in self.data_args["cat_list"]
        ]

        #Initilaizing the trackers for topics classifier
        self.topic_pred_xentropy = keras.metrics.Mean(name="topic_pred_x")
        self.topic_valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="topic_valid_acc")
    
    def compile(self, optimizer):
        super(TransformerClassifier, self).compile()
        self.optimizer = optimizer 
    
    def get_sentiment_pred_prob(self,idx_train,mask_train,cidx):     
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
        
        #Now we will apply the dense layer for this category
        cat_class_prob = self.cat_classifier_list[cidx](avg_embedding)

        return cat_class_prob
    
    def get_topic_pred_prob(self,idx_train,mask_train,cidx): 
        # raise NotImplementedError   
        #Getting the bert activation
        bert_outputs=self.bert_model(
                    input_ids=idx_train,
                    attention_mask=mask_train
        )
        bert_seq_output = bert_outputs.last_hidden_state

        #Multiplying this embedding with corresponding weights
        topic_imp_weights = tf.sigmoid(self.topic_importance_weight_list[cidx])
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
        
        #Now we will apply the dense layer for this topic
        topic_class_prob = self.topic_classifier_list[cidx](avg_embedding)

        return topic_class_prob
    
    def train_step(self,data):
        '''
        This function will run one step of training for the classifier
        '''
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        #Get the dataset for each category
        cat_dataset_list = data

        #Keeping track of classification accuracy
        cat_accuracy_list = []

        #Now getting the classification loss one by one each category        
        cat_total_loss = 0.0
        for cidx,cat in enumerate(self.data_args["cat_list"]):
            cat_label = cat_dataset_list[cat]["label"]
            cat_idx = cat_dataset_list[cat]["input_idx"]
            cat_mask = cat_dataset_list[cat]["attn_mask"]

            #Taking aside a chunk of data for validation
            valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

            #Getting the train data
            cat_label_train = cat_label[0:valid_idx]
            cat_idx_train = cat_idx[0:valid_idx]
            cat_mask_train = cat_mask[0:valid_idx]

            #Getting the validation data
            cat_label_valid = cat_label[valid_idx:]
            cat_idx_valid = cat_idx[valid_idx:]
            cat_mask_valid = cat_mask[valid_idx:]

            with tf.GradientTape() as tape:
                #Forward propagating the model
                cat_train_prob = self.get_sentiment_pred_prob(cat_idx_train,cat_mask_train,cidx)

                #Getting the loss for this classifier
                cat_loss = scxentropy_loss(cat_label_train,cat_train_prob)
                cat_total_loss += cat_loss
        
            #Now we have total classification loss, lets update the gradient
            cat_trainable_weights=[]
            for cidx in range(len(self.data_args["cat_list"])):
                cat_trainable_weights.append(
                                self.cat_importance_weight_list[cidx]
                )
                cat_trainable_weights += self.cat_classifier_list[cidx].trainable_weights
                
                
            grads = tape.gradient(cat_loss,cat_trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,cat_trainable_weights)
            )

            #Getting the validation accuracy for this category
            cat_valid_prob = self.get_sentiment_pred_prob(cat_idx_valid,cat_mask_valid,cidx)
            self.sent_valid_acc_list[cidx].update_state(cat_label_valid,cat_valid_prob)
        
        #Updating the metrics to track
        self.sent_pred_xentropy.update_state(cat_total_loss)



        #Now we will train the topics for the dataset
        all_topic_data = [(
                            cat_dataset_list[cat]["topic"],
                            cat_dataset_list[cat]["input_idx"],
                            cat_dataset_list[cat]["attn_mask"]
                    )
                        for cat in self.data_args["topic_list"]
        ]
        #Shuffling the topics for now (since one cat is ont topic for now)
        topic,input_idx,attn_mask = zip(*all_topic_data)
        topic = tf.random.shuffle(tf.concat(topic,axis=0))
        input_idx = tf.random.shuffle(tf.concat(input_idx,axis=0))
        attn_mask=tf.random.shuffle(tf.concat(attn_mask,axis=0))

        #Training the topic classifier in batches
        topic_total_loss = 0.0
        for tidx,topic in enumerate(self.data_args["topic_list"]):
            #Sharding the topic data into batches
            batch_size = self.data_args["batch_size"]
            topic_label = topic[tidx*batch_size:(tidx+1)*batch_size]
            topic_idx = input_idx[tidx*batch_size:(tidx+1)*batch_size]
            topic_mask = attn_mask[tidx*batch_size:(tidx+1)*batch_size]

            #Taking aside a chunk of data for validation
            valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

            #Getting the train data
            topic_label_train = topic_label[0:valid_idx]
            topic_idx_train = topic_idx[0:valid_idx]
            topic_mask_train = topic_mask[0:valid_idx]

            #Getting the validation data
            topic_label_valid = topic_label[valid_idx:]
            topic_idx_valid = topic_idx[valid_idx:]
            topic_mask_valid = topic_mask[valid_idx:]

            with tf.GradientTape() as tape:
                #Forward propagating the model
                topic_train_prob = self.get_topic_pred_prob(topic_idx_train,topic_mask_train,0)

                #Getting the loss for this classifier
                topic_loss = scxentropy_loss(topic_label_train,topic_train_prob)
                topic_total_loss += topic_loss
        
            #Now we have total classification loss, lets update the gradient
            topic_trainable_weights = [
                                            self.topic_importance_weight_list[0],
            ]
            topic_trainable_weights += self.topic_classifier_list[0].trainable_weights
            grads = tape.gradient(topic_loss,topic_trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,topic_trainable_weights)
            )

            #Getting the validation accuracy for this category
            topic_valid_prob = self.get_topic_pred_prob(topic_idx_valid,topic_mask_valid,0)
            self.topic_valid_acc.update_state(topic_label_valid,topic_valid_prob)
        
        #Updating the metrics to track
        self.topic_pred_xentropy.update_state(topic_total_loss)


        #Getting the tracking results
        track_dict= {
                    "sent_valid_acc_"+cat:self.sent_valid_acc_list[cidx].result()
                        for cidx,cat in enumerate(self.data_args["cat_list"])
        }
        track_dict["topic_valid_acc"]=self.topic_valid_acc.result()
        track_dict["xentropy_sent"]=self.sent_pred_xentropy.result()
        track_dict["xentorpy_topic"]=self.topic_pred_xentropy.result()

        return track_dict


def transformer_trainer(data_args,model_args):
    '''
    '''
    #First of all creating the model
    classifier = TransformerClassifier(data_args,model_args)

    #Now we will compile the model
    classifier.compile(
        keras.optimizers.Adam(learning_rate=model_args["lr"])
    )

    #Creating the dataset object
    data_handler = DataHandleTransformer(data_args)
    dataset = data_handler.amazon_reviews_handler()

    checkpoint_path = "{}/cp.ckpt".format(model_args["expt_name"])
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


    #Now fitting the model
    classifier.fit(
                        dataset,
                        epochs=model_args["epochs"],
                        callbacks=[cp_callback]
                        # validation_split=model_args["valid_split"]
    )


if __name__=="__main__":
    #Defining the Data args
    data_args={}
    data_args["path"] = "dataset/amazon/"
    data_args["transformer_name"]="bert-base-uncased"
    data_args["num_class"]=2
    data_args["max_len"]=200
    data_args["num_sample"]=100
    data_args["batch_size"]=8
    data_args["shuffle_size"]=data_args["batch_size"]*3
    data_args["cat_list"]=["arts","books","phones","clothes","groceries","movies","pets","tools"]
    data_args["topic_list"]=data_args["cat_list"]
    

    #Defining the Model args
    model_args={}
    model_args["expt_name"]="1.0"
    model_args["lr"]=0.001
    model_args["epochs"]=5
    model_args["valid_split"]=0.2
    model_args["train_bert"]=False
    model_args["bemb_dim"] = 768        #The dimension of bert produced last layer

    transformer_trainer(data_args,model_args)


            


