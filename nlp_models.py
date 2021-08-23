import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pdb
import os
import sys
from pprint import pprint


from nlp_data_handle import *

class SimpleBOW():
    '''
    This class will implement BiGRU/BiLSTM layers.
    '''
    def __init__(self,model_args):
        self.model_args = model_args
        self.data_handle = model_args["data_handle"]
    
    def _get_embedding_layer(self,embedding_matrix,max_len,trainable):
        '''
        This function will create our embedding layer initiaize with the
        corresponding weights.
        '''
        #Retreiving the mebdding dimensions
        vocab_len,emb_dim=embedding_matrix.shape
        #Creating the embeddign layer object
        embedding_layer=layers.Embedding(vocab_len,emb_dim,
                                        input_length=max_len,
                                        trainable=trainable
        )

        #Now building the layer by setting up the emb weights
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])

        return embedding_layer
    
    def get_bow_predictor(self,train,epochs):
        '''
        This class will implement the bag of words model for classification
        '''
        #Defining the input layer
        input_shape = (
                        self.data_handle.data_args["max_len"],
        )
        X_input = layers.Input(shape=input_shape,dtype="int32")


        #Taking the average correctly
        class BOWAvgLayer(keras.layers.Layer):
            def __init__(self,model_self):
                super(BOWAvgLayer,self).__init__()

                self.normalize_emb = model_self.model_args["normalize_emb"]

                #Getting the embedding layer
                self.embeddingLayerMain = model_self._get_embedding_layer(
                                                embedding_matrix=model_self.data_handle.emb_matrix,
                                                max_len=model_self.data_handle.data_args["max_len"],
                                                trainable=model_self.model_args["emb_train"],
                )

                #Getting the weight layer ofr each word
                vocab_weight_matrix = np.random.randn(len(model_self.data_handle.dict_w2i),1)
                self.embeddingLayerWeight = model_self._get_embedding_layer(
                                                embedding_matrix=vocab_weight_matrix,
                                                max_len=model_self.data_handle.data_args["max_len"],
                                                trainable=True,
                )
            
            def call(self,X_input):
                '''
                X_inputs: This is the word --> idx 
                X_emb   : This is the idx --> emb vectors
                '''
                #Creating our own mask and get number of non zero
                num_words=tf.cast(tf.math.logical_or(
                        tf.not_equal(X_input,0),
                        tf.not_equal(X_input,1),
                    ),
                    dtype=tf.float32,
                )
                num_words=tf.reduce_sum(num_words,axis=1,keepdims=True)

                #Now we will get the embedding of the inputs
                X_emb = self.embeddingLayerMain(X_input)
                #Normalizing the inputs
                if self.normalize_emb:
                    X_emb_norm = tf.math.l2_normalize(X_emb,axis=-1)
                else:
                    X_emb_norm = X_emb

                #Getting the weights of the individual words
                X_weight = self.embeddingLayerWeight(X_input)

                X_emb_weighted = X_emb_norm * tf.sigmoid(X_weight)

                #Now we need to take average of the embedding (zero vec non-train)
                X_bow=tf.divide(tf.reduce_sum(X_emb_weighted,axis=1,name="word_sum"),num_words)

                return X_bow
                
        bowAvgLayer = BOWAvgLayer(self)
        X_bow = bowAvgLayer(X_input)

        #Finally projecting to output layer
        prediction=layers.Dense(2,activation="softmax")(X_bow)

        #Compiling the model
        self.predictor=keras.Model(inputs=X_input,outputs=prediction)

        print(self.predictor.summary())

        if not train:
            return

        #Now we will train the model
        print("###########################################")
        print("######### Training the predictor ##########")
        print("###########################################")

        self.predictor.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )


        train_Y,train_X = zip(*self.data_handle.train_data)
        valid_Y,valid_X = zip(*self.data_handle.valid_data)
        print("NUmber of Training Instance:\t", len(train_Y))
        print("Number of Validation Instance:\t",len(valid_Y))
        # pdb.set_trace()
        self.predictor.fit(
                train_X,train_Y,
                epochs=epochs,
                validation_data=(valid_X,valid_Y),
                sample_weight=self.data_handle.sample_weight
        )

        
        #Now we will get the list of top important words.
        if self.model_args["save_imp"]:
            print("Saving the importance weights!")
            imp_weight_idx= 1 if model_args["emb_train"] else 0
            vocab_weights = np.squeeze(tf.sigmoid(bowAvgLayer.get_weights()[imp_weight_idx]).numpy())

            sorted_idx = np.argsort(vocab_weights).tolist()
            sorted_weights = vocab_weights[sorted_idx].tolist()
            # pdb.set_trace()

            sorted_words = [self.data_handle.dict_i2w[sidx].encode('utf-8') for sidx in sorted_idx]
            combined_importance = list(zip(sorted_words,sorted_weights))
            self._save_word_embedding(
                                        combined_importance,
                                        "importance_{}.tsv".format(self.model_args["expt_num"]),
                                        fmt=("%s %s"),
            )
        # pdb.set_trace()

        #Saving the embeddings for analysis
        if self.model_args["save_emb"]:
            print("Saving the learnt embedding")
            vocab_weights = np.squeeze(bowAvgLayer.get_weights()[1])
            emb_matrix = bowAvgLayer.get_weights()[0]
            word_name = [self.data_handle.dict_i2w[idx].encode('utf-8') for idx in range(len(self.data_handle.dict_i2w))]
            self._save_word_embedding(emb_matrix,"emb_matrix_{}.tsv".format(self.model_args["expt_num"]))
            self._save_word_embedding(vocab_weights,"weight_{}.tsv".format(self.model_args["expt_num"]))
            self._save_word_embedding(word_name,"name_{}.tsv".format(self.model_args["expt_num"]),fmt="%s")
  
    def _save_word_embedding(self,matrix,fname,fmt="%.6e"):
        '''
        '''
        #Saving the embedding matrix
        fpath ="embeddings/"+self.model_args["expt_folder"]+"/"
        os.makedirs(fpath,exist_ok=True)
        np.savetxt(fpath+fname,matrix,fmt=fmt,delimiter="\t")



if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-expt_num',dest="expt_num",type=str)
    parser.add_argument('-emb_path',dest="emb_path",type=str)
    parser.add_argument('-emb_train',dest="emb_train",type=bool)
    parser.add_argument('-normalize_emb',dest="normalize_emb",type=bool)
    parser.add_argument('-num_samples',dest="num_samples",type=int)
    parser.add_argument('-num_domains',dest="num_domains",type=int)
    args=parser.parse_args()
    print(args)


    #Creating the data handler
    data_args={}
    data_args["max_len"]=200        
    data_args["emb_path"]=args.emb_path#"glove-wiki-gigaword-100" #random  or glove-wiki-gigaword-100
    data_args["emb_dim"]=100
    data_args["train_split"]=0.8
    data_args["epsilon"] = 1e-3         #for numerical stability in sample weights
    data_args["num_samples"]=int(args.num_samples/args.num_domains)
    data_args["num_domains"] = args.num_domains
    data_handle = DataHandler(data_args)

    #Now creating our dataset from domain1 (original sentiment)
    # domain1_path = "counterfactually-augmented-data-master/sentiment/orig/"
    # X_D1,Y_D1 = data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)
    # pdb.set_trace()

    #Getting the data from both the domain
    # both_path = "counterfactually-augmented-data-master/sentiment/combined/"
    # X_D,Y_D = data_handle.data_handler_ltdiff_paper_sentiment(both_path)

    # #Initialize the embedding matrix
    # data_handle.load_embedding_mat()

    #Now we will start the training of basic model
    model_args={}
    model_args["data_handle"]=data_handle
    model_args["expt_folder"] = args.expt_num
    model_args["expt_num"] = args.expt_num + "."+ str(args.num_domains)  #single or both
    model_args["save_emb"] = False 
    model_args["save_imp"] = True
    model_args["emb_train"] = args.emb_train
    model_args["train_prop"] = 0.80     #80-20 split in the all dataset
    model_args["normalize_emb"] = args.normalize_emb


    sysout = sys.stdout
    syserr = sys.stderr
    log_fname = "logs/{}.txt".format(model_args["expt_num"])
    with open(log_fname,"w") as log_port:
        #Conneting the print pipe to log port
        sys.stdout = log_port
        sys.stderr = log_port

        if "amzn" in model_args["expt_num"]:
            #Defining the possible domains
            all_domain_list = ["arts","books","phones","clothes","groceries","movies","pets","tools"]
            # all_domain_list = ["beauty","software","appliance","faishon","giftcard","magazine"]
            domain_list = all_domain_list[0:data_args["num_domains"]]

            #Now generating the dataset according to the 
            path = "dataset/amazon/"
            data_handle.data_handler_amazon_reviews(path,domain_list,data_args["num_samples"])

        elif "fd1" in model_args["expt_num"]:
            #Getting the data from domain 1
            domain1_path = "counterfactually-augmented-data-master/sentiment/orig/eighty_percent/"
            Xtrain_D1,Xvalid_D1 = data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)

            if "both" in model_args["expt_num"]:
                #Getting the data from domain 2
                domain2_path = "counterfactually-augmented-data-master/sentiment/new/"
                Xtrain_D2,Xvalid_D2 = data_handle.data_handler_ltdiff_paper_sentiment(domain2_path)

                #Combining the dataset into one
                extra_prop = int((model_args["train_prop"] - 0.5) * len(Xvalid_D1))
                Xtrain_all_D1 = Xtrain_D1 + Xvalid_D1[0:extra_prop]

                Xtrain_all =  Xtrain_all_D1 + Xtrain_D2
                Xvalid_all = Xvalid_D1[extra_prop:] + Xvalid_D2

                print("Num of train from D1:\t",len(Xtrain_all_D1))
                print("Num of train from D2:\t",len(Xtrain_D2))
                print("Num of valid from D1:\t",len(Xvalid_D1[extra_prop:]))
                print("Num of valid from D2:\t",len(Xvalid_D2))
                

                #Now creating the sample weight
                D1_by_D2 = len(Xtrain_all_D1)/len(Xtrain_D2) 
                sample_weight = [1]*len(Xtrain_all_D1) + [D1_by_D2]*len(Xtrain_D2)
                print("sample weight for D1:\t",1)
                print("sample weight for D2:\t",D1_by_D2)

                data_handle.train_data = Xtrain_all
                data_handle.valid_data = Xvalid_all
                data_handle.sample_weight = sample_weight
            else:
                extra_prop = int((model_args["train_prop"] - 0.5) * len(Xvalid_D1))
                Xtrain_all_D1 = Xtrain_D1 + Xvalid_D1[0:extra_prop]

                Xtrain_all =  Xtrain_all_D1
                Xvalid_all = Xvalid_D1[extra_prop:]

                data_handle.train_data = Xtrain_all
                data_handle.valid_data = Xvalid_all

        elif "single" in model_args["expt_num"]:
            domain1_path = "counterfactually-augmented-data-master/sentiment/orig/"
            Xtrain_D1,Xvalid_D1 = data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)
        elif "both" in model_args["expt_num"]:
            both_path = "counterfactually-augmented-data-master/sentiment/combined/"
            Xtrain_D,Xvalid_D = data_handle.data_handler_ltdiff_paper_sentiment(both_path)
        else:
            raise NotImplementedError()
        
        #Initialize the embedding matrix
        data_handle.load_embedding_mat()


        simpleBOW = SimpleBOW(model_args)
        simpleBOW.get_bow_predictor(train=True,epochs=30)


