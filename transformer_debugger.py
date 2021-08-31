import numpy as np
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

        #Initializing the heads for each classifier
        self.cat_classifier_list = []
        for cat in self.data_args["cat_list"]:
            #Creating a dense layer
            cat_dense = layers.Dense(2,activation="softmax")
            self.cat_classifier_list.append(cat_dense)
        

        #Initializing the trackers
        self.pred_xentropy = keras.metrics.Mean(name="pred_x")
        self.pred_acc_list = [
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
        
        total_loss = 0.0
        for cidx,cat in enumerate(self.data_args["cat_list"]):
            with tf.GradientTape() as tape:
                cat_label,cat_idx,cat_mask = cat_dataset_list[cidx]

                #Getting the bert activation
                bert_outputs=self.bert_model(
                            input_ids=cat_idx,
                            attention_mask=cat_mask
                )
                bert_seq_output = bert_outputs.last_hidden_state

                #Now we need to take average of the last hidden state:
                #Dont use function, the output shape is not defined
                m = tf.cast(cat_mask,dtype=tf.float32)

                masked_sum =tf.reduce_sum(
                                bert_seq_output * tf.expand_dims(m,-1),
                                axis=1
                )

                num_tokens = tf.reduce_sum(m,axis=1,keepdims=True)

                avg_embedding = masked_sum / (num_tokens+1e-10)
                
                #Now we will apply the dense layer for this category
                cat_class_prob = self.cat_classifier_list[cidx](avg_embedding)

                #Getting the loss for this classifier
                cat_loss = scxentropy_loss(cat_label,cat_class_prob)
                total_loss += cat_loss

                #Updating the prediciton accuracy of current category
                self.pred_acc_list[cidx].update_state(cat_label,cat_class_prob)
        
                #Now we have total classification loss, lets update the gradient
                grads = tape.gradient(cat_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )
        
        #Updating the metrics to track
        self.pred_xentropy.update_state(total_loss)

        #Getting the tracking results
        track_dict= {
                    "pred_acc_"+cat:self.pred_acc_list[cidx].result()
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
    data_args["cat_list"]=["arts","books"]#,"phones","clothes","groceries","movies","pets","tools"]
    

    #Defining the Model args
    model_args={}
    model_args["lr"]=0.001
    model_args["epochs"]=5
    model_args["valid_split"]=0.2

    transformer_trainer(data_args,model_args)


            


