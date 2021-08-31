import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import BertModel,DistilBertModel

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
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        #Initializing the heads for each classifier
        self.cat_classifier_list = []
        for cat in self.data_args["cat_list"]:
            #Creating a dense layer
            cat_dense = layers.Dense(2,activation="softmax")
            self.cat_classfier_list.append(cat_dense)
        

        #Initializing the trackers
        self.pred_xentropy = keras.metrics.Mean(name="pred_x")
        self.prec_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="{}_acc".format(cat))
                for cat in self.data_args["cat_list"]
        ]
    
    def compile(self, optimizer):
        super(TransformerClassifier, self).compile()
        self.optimizer = optimizer 
    
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
        with tf.GradientTape() as tape:
            total_loss = 0.0
            for cidx,cat in enumerate(self.data_args["cat_list"]):
                cat_label,cat_idx,cat_mask = cat_dataset_list[cidx]

                #Getting the bert activation
                bert_outputs=self.bert_model(
                            input_ids=cat_idx,
                            attention_mask=cat_mask
                )
                bert_seq_output = bert_outputs.last_hidden_state

                #Now we need to take average of the last hidden state
                def mask_reduce_mean(x,m):
                    masked_sum = tf.squeeze(tf.sum(
                                x * tf.expand_dims(m,-1),
                                dim=1
                    ))

                    num_tokens = tf.sum(cat_mask,dim=1,keepdims=True)

                    masked_avg = masked_sum / (num_tokens+1e-10)
                    return masked_avg
                
                avg_embedding = mask_reduce_mean(bert_seq_output,cat_mask)

                #Now we will apply the dense layer for this category
                cat_class_prob = self.cat_classifier_list[cidx]

                #Getting the loss for this classifier
                cat_loss = scxentropy_loss(cat_label,cat_class_prob)
                total_loss += cat_loss

                #Updating the prediciton accuracy of current category
                self.prec_acc_list[cidx].update_state(cat_label,cat_class_prob)

            #Updating the metrics to track
            self.pred_xentropy.update_state(total_loss)

        
        #Now we have total classification loss, lets update the gradient
        grads = tape.gradient(total_loss,self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads,self.trainable_weights)
        )

        #Getting the tracking results
        track_dict= {
                    cat:self.pred_acc_list[cidx].result()
                        for cidx,cat in enumerate(self.data_args["cat_list"])
        }
        track_dict["xentropy"]=self.pred_xentropy.result()

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


    #Now fitting the model
    classifier.fit(
                        dataset,
                        epochs=model_args["epochs"],
                        validation_split=model_args["valid_split"]
    )


if __name__=="__init__":
    #Defining the Data args
    data_args={}
    data_args["path"] = "dataset/amazon/"
    data_args["num_class"]=2
    data_args["max_len"]=200
    data_args["num_sample"]=8000
    data_args["batch_size"]=128
    data_args["shuffle_size"]=data_args["batch_size"]*3
    data_args["cat_list"]=["arts","books","phones","clothes","groceries","movies","pets","tools"]
    

    #Defining the Model args
    model_args={}
    model_args["lr"]=0.001
    model_args["epochs"]=5
    model_args["valid_split"]=0.2

    transformer_trainer(data_args,model_args)


            


