import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pdb
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

                #Getting the embedding layer
                self.embeddingLayerMain = model_self._get_embedding_layer(
                                                embedding_matrix=model_self.data_handle.emb_matrix,
                                                max_len=model_self.data_handle.data_args["max_len"],
                                                trainable=True,
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
                X_weight = self.embeddingLayerWeight(X_input)

                X_emb_weighted = X_emb * X_weight

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
        # pdb.set_trace()
        self.predictor.fit(
                train_X,train_Y,epochs=epochs,validation_data=(valid_X,valid_Y),
        )

        
        #Now we will get the list of top important words.
        # pdb.set_trace()
        vocab_weights = np.squeeze(bowAvgLayer.get_weights()[1])
        emb_matrix = bowAvgLayer.get_weights()[0]
        word_name = [self.data_handle.dict_i2w[idx].encode('utf-8') for idx in range(len(self.data_handle.dict_i2w))]
        self._save_word_embedding(emb_matrix,"emb_matrix.tsv")
        self._save_word_embedding(vocab_weights,"weight.tsv")
        self._save_word_embedding(word_name,"name.tsv",fmt="%s")

    
    def _save_word_embedding(self,matrix,fname,fmt="%.6e"):
        '''
        '''
        #Saving the embedding matrix
        fpath ="embeddings/"+fname
        np.savetxt(fpath,matrix,fmt=fmt,delimiter="\t")






if __name__=="__main__":
    #Creating the data handler
    data_args={}
    data_args["max_len"]=100        
    data_args["emb_path"]="random"
    data_args["emb_dim"]=64
    data_handle = DataHandler(data_args)

    #Now creating our dataset from domain1 (original sentiment)
    domain1_path = "counterfactually-augmented-data-master/sentiment/orig/"
    data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)
    # pdb.set_trace()

    #Initialize the embedding matrix
    data_handle.load_embedding_mat()

    #Now we will start the training of basic model
    model_args={}
    model_args["data_handle"]=data_handle
    simpleBOW = SimpleBOW(model_args)
    simpleBOW.get_bow_predictor(train=True,epochs=25)


