import numpy as np
import pandas as pd
import scipy


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

            #Creating a dense layer
            topic_dense = layers.Dense(self.data_args["per_topic_class"],activation="softmax")
            self.topic_classifier_list.append(topic_dense)



        #Initializing the trackers for sentiment classifier
        self.sent_pred_xentropy = keras.metrics.Mean(name="sent_pred_x")
        self.sent_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="c{}_valid_acc".format(cat))
                for cat in self.data_args["cat_list"]
        ]

        #Initilaizing the trackers for topics classifier
        self.topic_pred_xentropy = keras.metrics.Mean(name="topic_pred_x")
        self.topic_valid_acc_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="t{}_valid_acc".format(topic))
                for topic in self.data_args["topic_list"]
        ]

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
    
    def get_topic_pred_prob(self,idx_train,mask_train,tidx): 
        # raise NotImplementedError   
        #Getting the bert activation
        bert_outputs=self.bert_model(
                    input_ids=idx_train,
                    attention_mask=mask_train
        )
        bert_seq_output = bert_outputs.last_hidden_state
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
        
        #Now we will apply the dense layer for this topic
        topic_class_prob = self.topic_classifier_list[tidx](avg_embedding)

        return topic_class_prob
    
    def train_step(self,data):
        '''
        This function will run one step of training for the classifier
        '''
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        cxentropy_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

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
            grads = tape.gradient(cat_loss,self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.trainable_weights)
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
                        for cat in self.data_args["cat_list"]
        ]
        #Shuffling the topics for now (since one cat is ont topic for now)
        topic,input_idx,attn_mask = zip(*all_topic_data)

        topic_label_all     =   tf.concat(topic,axis=0)
        input_idx           =   tf.concat(input_idx,axis=0)
        attn_mask           =   tf.concat(attn_mask,axis=0)

        #Shuffling the examples, incase the topic labels are skewed in a category
        if(self.model_args["shuffle_topic_batch"]):
            #To shuffle, dont use random shuffle independently
            num_samples = tf.shape(topic_label_all).numpy()[0]
            perm_indices = np.expand_dims(np.random.permutation(num_samples),axis=-1)

            #Now we will gather the tensors using this perm indices
            topic_label_all = tf.gather_nd(topic_label_all,indices=perm_indices)
            input_idx       = tf.gather_nd(input_idx,indices=perm_indices)
            attn_mask       = tf.gather_nd(attn_mask,indices=perm_indices)
        

        #Training the topic classifier in batches
        topic_total_loss = 0.0
        for iidx in range(len(self.data_args["cat_list"])):
            #Sharding the topic data into batches (to keep the batch size equal to topic ones)
            batch_size = self.data_args["batch_size"]
            topic_label = topic_label_all[iidx*batch_size:(iidx+1)*batch_size]
            topic_idx = input_idx[iidx*batch_size:(iidx+1)*batch_size]
            topic_mask = attn_mask[iidx*batch_size:(iidx+1)*batch_size]

            #Now we will iterate the training for each of the poic classifier
            for tidx,tname in enumerate(self.data_args["topic_list"]):
                #Taking aside a chunk of data for validation
                valid_idx = int( (1-self.model_args["valid_split"]) * self.data_args["batch_size"] )

                tclass_factor = self.data_args["per_topic_class"]
                #Getting the train data
                topic_label_train = topic_label[0:valid_idx,tidx]
                topic_idx_train = topic_idx[0:valid_idx]
                topic_mask_train = topic_mask[0:valid_idx]

                #Getting the validation data
                topic_label_valid = topic_label[valid_idx:,tidx]
                topic_idx_valid = topic_idx[valid_idx:]
                topic_mask_valid = topic_mask[valid_idx:]

                with tf.GradientTape() as tape:
                    #Forward propagating the model
                    topic_train_prob = self.get_topic_pred_prob(topic_idx_train,topic_mask_train,tidx)

                    #Getting the loss for this classifier
                    topic_loss = cxentropy_loss(topic_label_train,topic_train_prob)
                    topic_total_loss += topic_loss
            
                #Now we have total classification loss, lets update the gradient
                #TODO: BERT Model weight shouldn't be updated here. Correct this (used tf.stop_grad)
                grads = tape.gradient(topic_loss,self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads,self.trainable_weights)
                )

                #Getting the validation accuracy for this category
                topic_valid_prob = self.get_topic_pred_prob(topic_idx_valid,topic_mask_valid,tidx)

                #For calculating the accuracy we need one single label instead of mixed prob.
                topic_label_valid_MAX = tf.argmax(topic_label_valid,axis=-1)
                self.topic_valid_acc_list[tidx].update_state(topic_label_valid_MAX,topic_valid_prob)
        
        #Updating the metrics to track
        self.topic_pred_xentropy.update_state(topic_total_loss)


        #Getting the tracking results
        track_dict={}
        track_dict["xentropy_sent"]=self.sent_pred_xentropy.result()
        track_dict["xentorpy_topic"]=self.topic_pred_xentropy.result()

        #Function to all the accuracy to tracker
        def add_accuracy_mertrics(track_dict,acc_prefix,list_name,acc_list):
            for ttidx,ttname in enumerate(self.data_args[list_name]):
                track_dict["{}_{}".format(acc_prefix,ttidx)]=acc_list[ttidx].result()
        
        #Adding the accuracies
        add_accuracy_mertrics(track_dict=track_dict,
                                acc_prefix="sent_vacc",
                                list_name="cat_list",
                                acc_list=self.sent_valid_acc_list
        )
        add_accuracy_mertrics(track_dict=track_dict,
                                acc_prefix="topic_vacc",
                                list_name="topic_list",
                                acc_list=self.topic_valid_acc_list
        )

        return track_dict


def transformer_trainer(data_args,model_args):
    '''
    '''
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
    dataset = data_handler.amazon_reviews_handler()

    checkpoint_path = "nlp_logs/{}/cp.ckpt".format(model_args["expt_name"])
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    #Loading the weight if we want to reuse the computation
    if model_args["load_weight"]!=None:
        load_path = "nlp_logs/{}/cp.ckpt".format(model_args["load_weight"])
        load_dir = os.path.dirname(load_path)

        classifier.load_weights(load_path)


    #Now fitting the model
    classifier.fit(
                        dataset,
                        epochs=model_args["epochs"],
                        callbacks=[cp_callback]
                        # validation_split=model_args["valid_split"]
    )


def get_sorted_rank_correlation(sent_weights,topic_weights,topic_list,policy):
    #Getting the spurious ranking (decreasing order)
    dim_std = np.std(sent_weights,axis=1)
    dim_spurious_rank = np.argsort(-1*dim_std)

    #Now calculating the correlaiton
    topic_correlation = []
    for tidx,tname in enumerate(topic_list):
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
        else:
            raise NotImplementedError()
        
    #Sorting the ranking
    topic_correlation.sort(key=lambda x:-x[0])
    print("\ncorrelation_policy: ",policy)
    mypp(topic_correlation)

    return topic_correlation


def load_and_analyze_transformer(data_args,model_args):
    #First of all creating the model
    classifier = TransformerClassifier(data_args,model_args)
    
    
    checkpoint_path = "nlp_logs/{}/cp.ckpt".format(model_args["expt_name"])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    
    classifier.load_weights(checkpoint_path)


    #Getting the variance in the diemnsion of weights
    sent_weights=[
        np.squeeze(tf.sigmoid(classifier.cat_importance_weight_list[cidx]).numpy())
                        for cidx in range(len(data_args["cat_list"]))
    ]
    sent_weights = np.stack(sent_weights,axis=1)

    #Now getting the importance weights of the topics
    topic_weights=[
        np.squeeze(tf.sigmoid(classifier.topic_importance_weight_list[tidx]).numpy())
                        for tidx in range(len(data_args["topic_list"]))
    ]

    #Printing the different correlation weights
    get_sorted_rank_correlation(sent_weights,topic_weights,data_args["topic_list"],"weightedtau")
    get_sorted_rank_correlation(sent_weights,topic_weights,data_args["topic_list"],"kendalltau")
    get_sorted_rank_correlation(sent_weights,topic_weights,data_args["topic_list"],"spearmanr")


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
    parser.add_argument('-tfreq_ulim',dest="tfreq_ulim",type=float,default=1.0)
    parser.add_argument('-transformer',dest="transformer",type=str,default="bert-base-uncased")
    parser.add_argument('-num_epochs',dest="num_epochs",type=int)
    parser.add_argument("-load_weight",dest="load_weight",type=str,default=None)

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
    model_args["load_weight"]=args.load_weight
    model_args["lr"]=0.001
    model_args["epochs"]=args.num_epochs
    model_args["valid_split"]=0.2
    model_args["train_bert"]=args.train_bert
    model_args["bemb_dim"] = 768        #The dimension of bert produced last layer
    model_args["shuffle_topic_batch"]=False

    #Creating the metadata folder
    meta_folder = "nlp_logs/{}".format(model_args["expt_name"])
    os.makedirs(meta_folder,exist_ok=True)

    transformer_trainer(data_args,model_args)
    load_and_analyze_transformer(data_args,model_args)


            


