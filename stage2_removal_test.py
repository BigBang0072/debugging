import numpy as np
from scipy.spatial import distance

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data 
from tensorflow.keras import layers

import pdb
from pprint import pprint
import matplotlib.pyplot as plt

from inlp_debias import get_rowspace_projection,get_projection_to_intersection_of_nullspaces

class INLPRemover():
    '''
    This class will be used to run the removal process using the null-space projection method
    and test our hypothesis that this will correctly remove only the topic we intend to remove.
    '''
    def __init__(self,args):
        '''
        '''
        self.args=args
    
    def generate_dataset_old(self,):
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
    
    def generate_dataset_2D(self,):
        '''
        Here we will test the following things in 2D
        1. Convergence of the individual topics in presence not-perfect correlation
        2. Removal and new topic direction convergence
        3. Removal with subsequent iteration.
        '''
        print("Generating the dataset")
        #Creating the first topic (main topic)
        x1 = np.random.uniform(low=-1,high=1,size=self.args["num_examples"])
        y1 = 1*(x1>=0.0)

        #Creating the second topic (corrupting wrt to the main topic labels)
        '''
        Design Choices:
        1. Have a linear correlation with the first topic
        2. Have non-linear correlation with the first topic
        '''
        #Creating the no-so-perfect correlation (by reversing the labels)
        label_noise = self.args["label_noise"]
        y2=y1.copy()
        y2[np.where(y1==1)[0][0:int(self.args["num_examples"]/2*label_noise)]]= 0
        y2[np.where(y1==0)[0][0:int(self.args["num_examples"]/2*label_noise)]]= 1
        print("Num example reversed:",int(self.args["num_examples"]/2*label_noise))
        print("Overall disagreement: ", np.sum(y2==y1)/y2.shape[0])
        #Now creating the perfect data based on this label as topic 2
        #Case 1: linear correaltion with the data
        x2 = x1.copy()
        x2[y2==0] = -1* np.abs(x2[y2==0])
        x2[y2==1] = +1* np.abs(x2[y2==1])
        #Case 2: Non linear
        #Case 3: Random correlation (i.e non-functional)

        #Creating the topic direction
        self.ref_all_topic_dirs = [ 
                            [1.0,0.0],
                            [1.0,0.0]
        ]


        #Getting the statistics of the points
        print("topic x1: num_pos:{}\tnum_neg:{}".format(np.sum(y1==1),np.sum(y1==0)))
        print("topic x2: num_pos:{}\tnum_neg:{}".format(np.sum(y2==1),np.sum(y2==0)))
        #Plotting to check this relationship
        if(self.args["viz_data"]):
            color = np.zeros((self.args["num_examples"],3))
            color[y1==1]=[0,255,0]
            color[y1==0]=[255,0,0]
            plt.scatter(x1,x2,c=color/255.0)
            plt.show()

            color = np.zeros((self.args["num_examples"],3))
            color[y2==1]=[0,255,0]
            color[y2==0]=[255,0,0]
            plt.scatter(x1,x2,c=color/255.0)
            plt.show()


        #Creating the dataset
        X = np.stack([x1,x2],axis=-1)
        main_Y = y1.copy()*0.0
        topic_Y = np.stack([y1,y2],axis=-1)
        self.args["num_topics"]=X.shape[-1]

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
        dataset = self.generate_dataset_2D()
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
        main_init_vacc = model.main_valid_acc.result() 
        
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
        
        #Saving the model
        self.model = model
        topic_init_vacc = {
                    tidx:model.topic_valid_acc_list[tidx].result() 
                        for tidx in range(self.args["num_topics"])
        }

        #Getting the direction correlation
        all_topic_dir = self.get_all_direction()
        all_topic_angles = self.get_topic_convergence_angle(
                                    conv_all_topic_dirs=all_topic_dir,
                                    ref_all_topic_dirs=self.ref_all_topic_dirs,
        )
        return all_topic_angles


  

        pdb.set_trace()
        #Now removing the topic information one by one and checking the validation accuracy
        for tidx in range(self.args["num_topics"]):
            print("\nRemoving the topic : {}".format(tidx))

            #Getting the projection matrix
            topic_W_matrix = model.topic_classifier_list[tidx].get_weights()[0].T
            #Now getting the projection matrix
            P_W_curr = get_rowspace_projection(topic_W_matrix)
            all_proj_matrix_list = [P_W_curr]
            #Getting the aggregate projection matrix
            P_W = get_projection_to_intersection_of_nullspaces(
                                            rowspace_projection_matrices=all_proj_matrix_list,
                                            input_dim=model.args["latent_space_dim"]
            )

            #Getting the validation accuracy for main and this topic
            model.reset_all_metrics()
            #Lets see how this removal affects other topics also
            for other_tidx in range(self.args["num_topics"]):
                for data_batch in dataset:
                    model.valid_step(
                            data_batch=data_batch,
                            tidx=other_tidx,
                            P_matrix=P_W
                    )
            
                print("main_init:{:0.2f}\tmain_final:{:0.2f}\ttopic:{:}\ttopic_init:{:0.2f}\ttopic_final:{:0.2f}".format(
                                                    main_init_vacc,
                                                    model.main_valid_acc.result(),
                                                    other_tidx,
                                                    topic_init_vacc[other_tidx],
                                                    model.topic_valid_acc_list[other_tidx].result()
                ))

        return model
    
    def get_all_direction(self,):
        #Now we will get all the direction vector for each of the topic
        main_task_dir = np.squeeze(self.model.main_task_classifier.trainable_weights[0].numpy())
        print("Main Direction:")
        pprint(main_task_dir/np.linalg.norm(main_task_dir))

        #Now getting all the topic direction and getting their cosine similarity with the main task
        all_topic_cosine = {}
        all_topic_dir = []
        for tidx in range(self.args["num_topics"]):
            topic_dir = np.squeeze(self.model.topic_classifier_list[tidx].trainable_weights[0].numpy())
            all_topic_dir.append(topic_dir)
            topic_dist = distance.cosine(main_task_dir,topic_dir)
            print("\nTopic:{} Direction".format(tidx))
            pprint(topic_dir/np.linalg.norm(topic_dir))

            all_topic_cosine[tidx]=topic_dist
        
        print("All topic cosine-distance")
        pprint(all_topic_cosine)

        return all_topic_dir
    
    def get_topic_convergence_angle(self,ref_all_topic_dirs,conv_all_topic_dirs):
        '''
        Currently assuming all the topic directions

        ref_all_topic_dirs      : direction from which to measure the angle
        '''
        conv_all_topic_angles = []
        for tidx in range(self.args["num_topics"]):
            actual_topic_dir = ref_all_topic_dirs[tidx]
            conv_topic_dir = conv_all_topic_dirs[tidx]

            #Getting the angle
            theta = np.arccos(
                                np.sum((actual_topic_dir*conv_topic_dir))\
                                    /(np.linalg.norm(actual_topic_dir)*np.linalg.norm(conv_topic_dir))
            )/np.pi
            print("topic:{}\tangle:{}".format(tidx,theta))
            conv_all_topic_angles.append(theta)
        
        return conv_all_topic_angles


class MyModel(keras.Model):
    '''
    This will the maain NN class which will be used to train the main classifier and the 
    other topic classifier.
    '''
    def __init__(self,args):
        super(MyModel,self).__init__()
        self.args=args

        #Creting the encoder layers
        self.encoder_layer_list=[]
        for lidx in range(self.args["num_hlayer"]):
            self.encoder_layer_list.append(
                        layers.Dense(self.args["hlayer_dim"],activation="relu")
            )
        if(self.args["num_hlayer"]!=0):
            self.args["latent_space_dim"]=self.args["hlayer_dim"]#for now
        else:
            self.args["latent_space_dim"]=self.args["num_topics"]#for now
        

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
    
    def pass_via_encoder(self,X):
        '''
        '''
        for hlayer in self.encoder_layer_list:
            X = hlayer(X)
        return X
    
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
                train_enc = self.pass_via_encoder(train_X)
                train_prob = self.main_task_classifier(train_enc)
                main_loss = bxentropy_loss(train_main_Y,train_prob)
            
            #Training the main task params
            params = []
            for hlayer in self.encoder_layer_list:
                params += hlayer.trainable_weights
            params += self.main_task_classifier.trainable_weights
            grads = tape.gradient(main_loss,params)
            self.optimizer.apply_gradients(
                zip(grads,params)
            )

            #Updating the loss
            self.main_xentropy_loss.update_state(main_loss)

            #Getting the validation accuracy
            if(valid_X.shape[0]!=0):
                valid_enc = self.pass_via_encoder(valid_X)
                valid_pred = self.main_task_classifier(valid_enc)
                self.main_valid_acc.update_state(valid_main_Y,valid_pred)

        elif task=="topic":
            #Training a particular topic classifier
            with tf.GradientTape() as tape:
                #Forward pass though the topic classifier
                train_enc = self.pass_via_encoder(train_X)
                train_pred = self.topic_classifier_list[tidx](train_enc)
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
                valid_enc = self.pass_via_encoder(valid_X)
                valid_pred = self.topic_classifier_list[tidx](valid_enc)
                self.topic_valid_acc_list[tidx].update_state(valid_topic_Y[:,tidx],valid_pred)

        else:
            raise NotImplementedError()
    
    def valid_step(self,data_batch,tidx,P_matrix):
        '''
        '''
        #Getting the dataset
        valid_idx = int( (1-self.args["valid_split"]) * self.args["batch_size"] )

        #Getting the valid dataset
        valid_X = data_batch["X"][valid_idx:]
        valid_main_Y = data_batch["main_Y"][valid_idx:]
        valid_topic_Y = data_batch["topic_Y"][valid_idx:]

        #Getting the validation accuracy on every task
        #Getting the encoded X
        valid_enc = self.pass_via_encoder(valid_X)
        X_proj = self._get_proj_X_enc(valid_enc,P_matrix)

        #Now getting the main task accuracy
        main_pred = self.main_task_classifier(X_proj)
        if(valid_X.shape[0]!=0):
            self.main_valid_acc.update_state(valid_main_Y,main_pred)

        #Now getting the topic task accuracy
        topic_pred = self.topic_classifier_list[tidx](X_proj)
        if(valid_X.shape[0]!=0):
            self.topic_valid_acc_list[tidx].update_state(valid_topic_Y[:,tidx],topic_pred)
   
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


if __name__=="__main__":
    #Getting the required arguments
    args = {}
    args["num_examples"]=1000
    args["batch_size"]=128
    args["valid_split"]=0.2
    args["viz_data"]=True 

    args["lr"]=0.005
    args["main_epochs"]=1
    args["topic_epochs"]=50

    args["num_hlayer"] = 0
    args["hlayer_dim"] = -1


    args["beta"] = 1.0
    args["label_noise"] = 1-0.5  #not-so-perfect correlatedness


    
    #Buying the 10 rs "remover"
    remover = INLPRemover(args)
    remover.trainer()

