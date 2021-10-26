import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data 
from tensorflow.keras import layers

import pdb
from pprint import pprint
import matplotlib.pyplot as plt



class INLPRemover():
    '''
    This class will be used to run the removal process using the null-space projection method
    and test our hypothesis that this will correctly remove only the topic we intend to remove.
    '''
    def __init__(self,args):
        '''
        '''
        self.args=args
    
    def generate_dataset(self,):
        '''
        '''
        #Causal Features/Topics
        x1 = 0.0 + 1.0*np.random.randn(self.args["num_examples"])
        
        #Monotonically correlated Feture/Topics
        x2 = 1*x1+10.0 #+noise
        x3 = 10*(x1**3) + 20.0 #+noise
        
        #Non-Monotonically correlated Feature/Topics
        x4 = 1.0/(1+10*(x1)**3) #+noise

        #Useless feaure
        x5 = 0.0+ 1.0*np.random.randn(self.args["num_examples"])

        #Collecting all the features
        X = np.stack([x1,x2,x3,x4,x5],axis=-1)
        #Creating the topic labels
        topic_Y = (X>np.mean(X,axis=0))*1.0
        #Getting the labels
        main_Y = (x1>np.mean(x1))*1.0

        #Adding the number of topics
        self.args["num_topics"]=X.shape[-1]

        #Plotting the results
        if self.args["viz_data"]==True:
            for tidx in range(X.shape[-1]):
                plt.hist(X[:,tidx],edgecolor="k",bins=30)
                plt.show()
        

        #Generating the dataset
        dataset = tf.data.Dataset.from_tensor_slices(
                                        dict(
                                            X = X.astype(np.float32),
                                            topic_Y = topic_Y.astype(np.int32),
                                            main_Y = main_Y.astype(np.int32)
                                        )
        )
        dataset = dataset.batch(self.args["batch_size"])

        return dataset
    
    def trainer(self,):
        '''
        '''
        #Creating the dataset
        dataset = self.generate_dataset()
        #Creating the model
        model = MyModel(self.args)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.args["lr"])
        )

        #Now we will start the main task training
        for enum in range(self.args["main_epochs"]):
            #Resetting the metrics
            model.reset_all_metrics()

            #Training the model for one full pass
            for data_batch in dataset:
                model.train_step(
                            data_batch=data_batch,
                            tidx=None,
                            task="main"
                )
            print("epoch:{:}\txentropy:{:0.3f}\tmain_vacc:{:0.2f}".format(
                                                        enum,
                                                        model.main_xentropy_loss.result(),
                                                        model.main_valid_acc.result()
            ))
        
        #Next we will train all the topic classifier
        print("Training the Topics!!")
        for enum in range(self.args["topic_epochs"]):
            #Resetting the metrics
            model.reset_all_metrics()
            print("###################################")
            print("Starting new Epoch!")

            #Traing each topic one by one
            for tidx in range(self.args["num_topics"]):
                for data_batch in dataset:
                    model.train_step(
                                data_batch=data_batch,
                                tidx=tidx,
                                task="topic"
                    )
            
                print("epoch:{:}\ttopic:{:}\txentropy:{:0.3f}\ttopic_vacc:{:0.2f}".format(
                                                    enum,
                                                    tidx,
                                                    model.topic_xentropy_loss_list[tidx].result(),
                                                    model.topic_valid_acc_list[tidx].result()
                ))


class MyModel(keras.Model):
    '''
    This will the maain NN class which will be used to train the main classifier and the 
    other topic classifier.
    '''
    def __init__(self,args):
        super(MyModel,self).__init__()
        self.args=args

        #Creating the main task classifier
        self.main_task_classifier = layers.Dense(1,activation="sigmoid")
        #Getting the loss metrics
        self.main_xentropy_loss = keras.metrics.Mean(name="main_xentropy")
        self.main_valid_acc = keras.metrics.BinaryAccuracy(name="main_vacc",threshold=0.5)

        #Creating the topic classifier
        self.topic_classifier_list=[]
        self.topic_xentropy_loss_list = []
        self.topic_valid_acc_list = []
        for tidx in range(self.args["num_topics"]):
            #Creating the topic classifier
            self.topic_classifier_list.append(
                        layers.Dense(1,activation="sigmoid")
            )

            #Creating the loss metrics for the topic
            self.topic_xentropy_loss_list.append(
                        keras.metrics.Mean(name="topic_{}_xentropy".format(tidx))
            )
            self.topic_valid_acc_list.append(
                        keras.metrics.BinaryAccuracy(
                                            name="topic_{}_vacc".format(tidx),
                                            threshold=0.5
                        )
            )
    
    def reset_all_metrics(self,):
        self.main_xentropy_loss.reset_state()
        self.main_valid_acc.reset_state()

        for tidx in range(self.args["num_topics"]):
            self.topic_xentropy_loss_list[tidx].reset_state()
            self.topic_valid_acc_list[tidx].reset_state()
    
    def compile(self,optimizer):
        super(MyModel, self).compile()
        self.optimizer = optimizer 
    
    def train_step(self,data_batch,tidx,task):
        '''
        '''
        #Getting the dataset
        valid_idx = int( (1-self.args["valid_split"]) * self.args["batch_size"] )

        #Getting the train dataset
        train_X = data_batch["X"][0:valid_idx]
        train_main_Y = data_batch["main_Y"][0:valid_idx]
        train_topic_Y = data_batch["topic_Y"][0:valid_idx]

        #Getting the valid dataset
        valid_X = data_batch["X"][valid_idx:]
        valid_main_Y = data_batch["main_Y"][valid_idx:]
        valid_topic_Y = data_batch["topic_Y"][valid_idx:]

        #Defining the loss function
        bxentropy_loss = keras.losses.BinaryCrossentropy(from_logits=False)

        if task=="main":
            with tf.GradientTape() as tape:
                #Making the forward pass
                train_prob = self.main_task_classifier(train_X)
                main_loss = bxentropy_loss(train_main_Y,train_prob)
            
            #Training the main task params
            grads = tape.gradient(main_loss,self.main_task_classifier.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.main_task_classifier.trainable_weights)
            )

            #Updating the loss
            self.main_xentropy_loss.update_state(main_loss)

            #Getting the validation accuracy
            if(valid_X.shape[0]!=0):
                valid_pred = self.main_task_classifier(valid_X)
                self.main_valid_acc.update_state(valid_main_Y,valid_pred)

        elif task=="topic":
            #Training a particular topic classifier
            with tf.GradientTape() as tape:
                #Forward pass though the topic classifier
                train_pred = self.topic_classifier_list[tidx](train_X)
                topic_loss = bxentropy_loss(train_topic_Y[:,tidx],train_pred)
            
            #Training the topic classifier
            grads = tape.gradient(topic_loss,self.topic_classifier_list[tidx].trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads,self.topic_classifier_list[tidx].trainable_weights)
            )

            #Updating the topic loss
            self.topic_xentropy_loss_list[tidx].update_state(topic_loss)

            #Getting the validation accuracy of the topic
            if(valid_X.shape[0]!=0):
                valid_pred = self.topic_classifier_list[tidx](valid_X)
                self.topic_valid_acc_list[tidx].update_state(valid_topic_Y[:,tidx],valid_pred)

        else:
            raise NotImplementedError()


if __name__=="__main__":
    #Getting the required arguments
    args = {}
    args["num_examples"]=10000
    args["batch_size"]=128
    args["valid_split"]=0.2
    args["viz_data"]=False 

    args["lr"]=0.005
    args["main_epochs"]=10
    args["topic_epochs"]=10


    
    #Buying the 10 rs "remover"
    remover = INLPRemover(args)
    remover.trainer()

