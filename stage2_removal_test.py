from re import search
from typing import DefaultDict
import numpy as np
from scipy.spatial import distance
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data 
from tensorflow.keras import layers

import pdb
from tqdm import tqdm
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
        # point_density = 2.0/self.args["num_example"]
        # actual_mean = 0.5

        #If we want to inject some inherent noise in the first topic itself
        if("inject_noise" in self.args and self.args["inject_noise"]==True):
            # noise_level = self.args["inject_noise"]
            # num_reverse = int(self.args["num_exmaples"]//2)
            # mval = self.args["noise_pos"]
            # noise_ratio = self.args["noise_ratio"]

            # noise_mean  = mval * actual_mean  #self.args["noise_mean"]
            # noise_sigma =  noise_ratio/(1+noise_ratio) #self.args["noise_sigma"] #will control number of example

            # #Setting the m-value
            # self.args["mval"] = mval

            #Adding extra point

            noise_mean  = self.args["noise_mean"]
            noise_sigma =  self.args["noise_sigma"] #will control number of example

            # Reversing the positive labels
            positive_mask = np.logical_and(
                                x1>(noise_mean-noise_sigma),
                                x1<(noise_mean+noise_sigma)
            )
            y1[positive_mask]=0

            #Reversing the negative labels
            negative_mask = np.logical_and(
                                x1>(-1*noise_mean-noise_sigma),
                                x1<(-1*noise_mean+noise_sigma)
            )
            y1[negative_mask]=1

            print("Number of points flipped: pos:{}\tneg:{}".format( 
                                            np.sum(positive_mask),
                                            np.sum(negative_mask)
            ))

            #Adding the stats for the calculating the angle theoretically
            actual_point_mask = np.logical_and(x1>0,y1==1)
            actual_mean = np.mean(x1[actual_point_mask])
            self.args["mval"] = noise_mean/actual_mean 
            self.args["noise_ratio"] = np.sum(positive_mask)/np.sum(x1>=0.0)

            

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
        x2 = x2*0.0


        #Getting the statistics of the points
        print("topic x1: num_pos:{}\tnum_neg:{}".format(np.sum(y1==1),np.sum(y1==0)))
        print("topic x2: num_pos:{}\tnum_neg:{}".format(np.sum(y2==1),np.sum(y2==0)))
        #Plotting to check this relationship
        if(self.args["viz_data"]):
            self.plot_2D_dataset(x1,x2,y1,y2)


        #Creating the dataset
        topic_X = np.stack([x1,x2],axis=-1)
        topic_Y = np.stack([y1,y2],axis=-1)
        self.args["num_topics"]=topic_X.shape[-1]

        main_X = np.stack([x1,x2*0.0],axis=-1)
        main_Y = y1.copy()

        dataset = tf.data.Dataset.from_tensor_slices(
                                        dict(
                                            X = topic_X.astype(np.float32),
                                            main_X = main_X.astype(np.float32),
                                            topic_Y = topic_Y.astype(np.int32),
                                            main_Y = main_Y.astype(np.int32)
                                        )
        )
        dataset = dataset.batch(self.args["batch_size"])

        return dataset
    
    def generate_dataset_1D_point(self,):
        '''
        This will be used to test the convergence in presence of noise
        first using the point strategy and then matching with the theory.
        
        Next we will then test the results when we have a data distribution 
        instead of the point mass and then check the convergence.

        Next we will then test the scenrio that will happen after the projection
        of the 2D topic i.e mixing of fully predictive and the partially predictive
        resulting in again a noisy dataset.
        '''
        #Creating the split between noise and actual mechnism
        num_examples = self.args["num_examples"]
        num_mech_noise = int(num_examples*self.args["noise_ratio"])
        num_mech_act  = num_examples - num_mech_noise
        print("num_mech_noise:{}\tnum_mech_act{}".format(
                                                    num_mech_noise,
                                                    num_mech_act
        ))

        #Creating the first topic (main topic)
        m_val = self.args["noise_pos"]
        pov_pos = 2.0
        neg_pos = m_val * pov_pos
        self.args["mval"]=m_val
        x1 = np.array([-pov_pos,pov_pos]*num_mech_act + [-1.0*neg_pos,neg_pos]*num_mech_noise,dtype=np.float32)
        y1 = np.array([0,1]*num_mech_act + [1,0]*num_mech_noise, dtype=np.int32)
        # print(np.stack([x1,y1],axis=-1))
        #Shuffling the examples
        perm = np.random.permutation(2*self.args["num_examples"])
        x1,y1 = x1[perm],y1[perm]
        # print(np.stack([x1,y1],axis=-1)[-100:])

        #Creating the topic direction
        self.ref_all_topic_dirs = [ 
                            [1.0,0.0],
        ]

        #Dummy x2 variable
        x2 = x1.copy()*0.0
        y2 = y1.copy()

        #Cehcking the dataset stats in the validation set
        vratio = self.args["valid_split"]
        vstart = int(vratio*x1.shape[0])
        x1_valid = x1[-vstart:]
        y1_valid = y1[-vstart:]
        num_positive_labels = np.sum(y1_valid==1)
        num_actual_mech = np.sum(np.logical_and(x1_valid>0,y1_valid>0))
        print("num_valid:{}\tnratio:{}\tnum_positive:{}\tnum_actual_positive:{}".format(
                                                    x1_valid.shape[0],
                                                    self.args["noise_ratio"],
                                                    num_positive_labels,
                                                    num_actual_mech,
        ))

        #Checking overall statistics
        num_positive_labels = np.sum(y1==1)
        num_actual_mech = np.sum(np.logical_and(x1>0,y1>0))
        print("num_all:{}\tnratio:{}\tnum_positive:{}\tnum_actual_positive:{}".format(
                                                    x1.shape[0],
                                                    self.args["noise_ratio"],
                                                    num_positive_labels,
                                                    num_actual_mech,
        ))
        # pdb.set_trace()



        #Getting the statistics of the points
        print("topic x1: num_pos:{}\tnum_neg:{}".format(np.sum(y1==1),np.sum(y1==0)))
        #Plotting to check this relationship
        if(self.args["viz_data"]):
            self.plot_2D_dataset(x1,x1*0.0,y1,y1)


        #Creating the dataset
        topic_X = np.stack([x1,x2],axis=-1)
        topic_Y = np.stack([y1,y2],axis=-1)
        self.args["num_topics"]=topic_X.shape[-1]

        main_X = np.stack([x1,x2*0.0],axis=-1)
        main_Y = y1.copy()

        dataset = tf.data.Dataset.from_tensor_slices(
                                        dict(
                                            X = topic_X.astype(np.float32),
                                            main_X = main_X.astype(np.float32),
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

        #Training the main task classifier  
        main_init_vacc = self.train_main_task_classifier(dataset,model,None) 
        
        #Next we will train all the topic classifier
        self.train_all_topic_classifiers(dataset,model,None)
        
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
    
    def trainer_with_inlp(self,):
        '''
        '''
        #Creating the dataset
        dataset = self.generate_dataset_2D()
        #Creating the model
        model = MyModel(self.args)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.args["lr"])
        )
        self.model=model
        #Getting the main task primed (not needed right now)
        main_init_vacc = self.train_main_task_classifier(dataset,model,None)
        topic_epochs = self.args["topic_epochs"]
        self.args["topic_epochs"]=1
        self.train_all_topic_classifiers(dataset,model,None)
        self.args["topic_epochs"]=topic_epochs
        init_weights = model.get_weights() 

        #Initializing the P_matrix
        P_matrix = None
        P_matrix_list = []
        #Next we will run the iterative null space projection
        # for inum in range(2):#self.args["num_inlp_iter"]):

        #TODO Resetting the final topic classifiers so that we dont inject initialization bias

        #First of all we will train the topic classifiers
        self.train_all_topic_classifiers(dataset,model,P_matrix)
        #Getting the convergence metrics
        all_topic_dir = self.get_all_direction()
        all_topic_angles = self.get_topic_convergence_angle(
                                    conv_all_topic_dirs=all_topic_dir,
                                    ref_all_topic_dirs=self.ref_all_topic_dirs,
        )
        print("Initial Topic Angle")
        print(all_topic_angles)
        self.get_angle_diff([0.0,0.5],all_topic_angles)

        #Getting the topic accuracy
        topic1_accuracy = model.topic_valid_acc_list[0].result()
        return all_topic_angles,topic1_accuracy







        print("\n\nStarting Removal Step!")
        #Next we will apply the nul space removal
        #Getting the projection matrix for topic 0
        topic_W_matrix = model.topic_classifier_list[self.args["remove_tidx"]].get_weights()[0].T
        #Now getting the projection matrix
        P_W_curr = get_rowspace_projection(topic_W_matrix)
        P_matrix_list.append(P_W_curr)
        #Getting the aggregate projection matrix
        P_matrix_1 = get_projection_to_intersection_of_nullspaces(
                                        rowspace_projection_matrices=P_matrix_list,
                                        input_dim=model.args["latent_space_dim"]
        )

        #Now we will gather the after-removal metrics
        iter2_topic_theory_direction = [
            np.array([np.sin(np.pi*all_topic_angles[0])   , -1*np.cos(np.pi*all_topic_angles[0])]),#(t(1) dir)
            np.array([-1*np.sin(np.pi*all_topic_angles[0]), np.cos(np.pi*all_topic_angles[0])]),#(s(1) dir)
        ]
        #Getting the new topic directions
        new_topic_theory_angles = self.get_topic_convergence_angle(
                                        conv_all_topic_dirs=iter2_topic_theory_direction,
                                        ref_all_topic_dirs=self.ref_all_topic_dirs
        )
        print("\n\nNew topic theoretical angles (*pi):")
        print(new_topic_theory_angles)

        #Getting the components in that directions
        # print("Component in new theoretical direction")
        # print("t(1): angle:{}\ttheory_comp:{}\t")

        #TODO Resetting the classifier to get rid of the old converged topic classifier
        # model.initialize_topic_related_machinery()
        # model = MyModel(self.args)
        # model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=self.args["lr"])
        # )
        # self.model=model
        # #Getting the main task primed (not needed right now)
        # main_init_vacc = self.train_main_task_classifier(dataset,model,None)

        #Vizualiing the projected data once
        self.get_projected_dataset(dataset,P_matrix_1)
        model.set_weights(init_weights)

        #Now retraining the topic classifier
        self.train_all_topic_classifiers(dataset,model,P_matrix_1)
        all_topic_dir = self.get_all_direction()
        all_topic_angles = self.get_topic_convergence_angle(
                                    conv_all_topic_dirs=all_topic_dir,
                                    ref_all_topic_dirs=self.ref_all_topic_dirs,
        )
        print("\n\nNew Converged topic angle (*pi):")
        print(all_topic_angles)
        self.get_angle_diff(new_topic_theory_angles,all_topic_angles)
    
    def trainer_convergence(self,):
        '''
        '''
        #Creating the dataset
        dataset = self.generate_dataset_1D_point()
        #Creating the model
        model = MyModel(self.args)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.args["lr"])
        )
        self.model=model
        #Getting the main task primed (not needed right now)
        main_init_vacc = self.train_main_task_classifier(dataset,model,None)
        topic_epochs = self.args["topic_epochs"]
        self.args["topic_epochs"]=1
        self.train_all_topic_classifiers(dataset,model,None)
        self.args["topic_epochs"]=topic_epochs
        init_weights = model.get_weights() 

        #Initializing the P_matrix
        P_matrix = None
        P_matrix_list = []
        #Next we will run the iterative null space projection
        # for inum in range(2):#self.args["num_inlp_iter"]):

        #TODO Resetting the final topic classifiers so that we dont inject initialization bias

        #First of all we will train the topic classifiers
        self.train_all_topic_classifiers(dataset,model,P_matrix)
        #Getting the convergence metrics
        all_topic_dir = self.get_all_direction()
        all_topic_angles = self.get_topic_convergence_angle(
                                    conv_all_topic_dirs=all_topic_dir,
                                    ref_all_topic_dirs=self.ref_all_topic_dirs,
        )
        print("Initial Topic Angle")
        print(all_topic_angles)
        self.get_angle_diff([0.0,0.5],all_topic_angles)

        #Getting the topic accuracy
        topic1_accuracy = model.topic_valid_acc_list[0].result()
        return all_topic_angles,topic1_accuracy
    
    def train_main_task_classifier(self,dataset,model,P_matrix):
        '''
        '''
        #Now we will start the main task training
        for enum in range(self.args["main_epochs"]):
            #Resetting the metrics
            model.reset_all_metrics()

            #Training the model for one full pass
            for data_batch in dataset:
                model.train_step(
                            data_batch=data_batch,
                            tidx=None,
                            task="main",
                            P_matrix=P_matrix,
                )
            print("epoch:{:}\txentropy:{:0.3f}\tmain_vacc:{:0.2f}".format(
                                                        enum,
                                                        model.main_xentropy_loss.result(),
                                                        model.main_valid_acc.result()
            ))
        main_init_vacc = model.main_valid_acc.result() 

    def train_all_topic_classifiers(self,dataset,model,P_matrix):
        '''
        '''
        print("Training the Topics!!")
        tbar = tqdm(range(self.args["topic_epochs"]))
        for enum in tbar:
            #Resetting the metrics
            model.reset_all_metrics()
            # print("###################################")
            # print("Starting new Epoch!")

            #Traing each topic one by one
            for tidx in range(self.args["num_topics"]):
                for data_batch in dataset:
                    model.train_step(
                                data_batch=data_batch,
                                tidx=tidx,
                                task="topic",
                                P_matrix=P_matrix
                    )
            
                # tbar.set_postfix_str("epoch:{:}\ttopic:{:}\txentropy:{:0.3f}\ttopic_vacc:{:0.2f}".format(
                #                                     enum,
                #                                     tidx,
                #                                     model.topic_xentropy_loss_list[tidx].result(),
                #                                     model.topic_valid_acc_list[tidx].result()
                # ))

            tbar.set_postfix_str("epoch:{:}, topic0_vacc:{:0.2f}, topic1_vacc:{:0.2f}".format(
                                                enum,
                                                model.topic_valid_acc_list[0].result(),
                                                model.topic_valid_acc_list[1].result()
            ))

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
            pprint(topic_dir)

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
        for tidx in range(len(ref_all_topic_dirs)):
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

    def get_angle_diff(self,angle_list_ref,angle_list_compare):
        '''
        '''
        print("Getting angle diff list:")
        diff_list=[]
        for aidx,(angle1,angle2) in enumerate(zip(angle_list_ref,angle_list_compare)):
            print("aidx:{}\taref:{}\tacomp:{}\tdiff:{}".format(
                            aidx,
                            angle1*180.0,
                            angle2*180.0,
                            (angle2-angle1)*180.0,
            ))
            diff_list.append((angle2-angle1)*180.0)
        
        return diff_list
    
    def get_projected_dataset(self,dataset,P_matrix):
        '''
        '''
        X_list,topic_Y_list= [],[]
        for data_batch in dataset:
            X,Y=self.model.get_projected_dataset(data_batch,P_matrix)
            X_list.append(X)
            topic_Y_list.append(Y)
        
        #Getting the full dataset
        X = np.concatenate(X_list)
        topic_Y = np.concatenate(topic_Y_list)

        #Plotting the dataset
        if(self.args["viz_data"]):
            self.plot_2D_dataset(X[:,0],X[:,1],topic_Y[:,0],topic_Y[:,1])
        
    def plot_2D_dataset(self,x1,x2,y1,y2):
        '''
        '''
        #Sampling the points
        perm = np.random.permutation(np.arange(0,x1.shape[0]))[0:100]
        x1=x1[perm]
        x2=x2[perm]
        y1=y1[perm]
        y2=y2[perm]


        color = np.zeros((y1.shape[0],3))
        color[y1==1]=[0,255,0]
        color[y1==0]=[255,0,0]
        plt.scatter(x1,x2,c=color/255.0,alpha=0.5)
        plt.xlim(-1.5,1.5)
        plt.show()

        color = np.zeros((y2.shape[0],3))
        color[y2==1]=[0,255,0]
        color[y2==0]=[255,0,0]
        plt.scatter(x1,x2,c=color/255.0,alpha=0.5)
        plt.xlim(-1.5,1.5)
        plt.show()
    
    def get_converged_angle_with_noise_theory(self,):
        '''
        This will calculate the location where the angle should converge based
        on the noise position and the noise ratio.
        '''
        #Getting the alpha factor (alpha_major/alpha_minor)
        alpha_minor = self.args["noise_ratio"]
        alpha_major = 1.0-alpha_minor
        alpha = alpha_major/alpha_minor #malfoy manor :)

        #Getting the relative position value (resacale if noise pos)
        m = self.args["mval"]
        #Assert this when we are having the acual pos as 1.0
        # assert self.args["noise_pos"]==self.args["mval"]

        #getting the search value
        search_val = alpha/m 

        #Checking if roots are possible or not
        LB = (1+np.exp(-1))/(1+np.exp(m))
        UB = (1+np.exp(1))/(1+np.exp(-m))
        print("LB:{}\t UB:{}\t alpha/m:{}".format(LB,UB,search_val))
        assert LB<=UB,"Bound Error"

        if(search_val<LB):
            print("No root exist cuz alpha/m less than LB")
            print("Convergence angle: {}pi".format(1))
            return np.pi 
        elif (search_val>UB):
            print("No root exist cuz alpha/m greater than UB")
            print("Convergence angle: {}pi".format(0))
            return 0.0
        elif (search_val<1):
            print("Root exist")
            print("Convergence in [pi/2,pi]")
        elif (search_val>1):
            print("Root Exist")
            print("Convergence in [0,pi/2]")
        
        

        #Now binary searching the correction convergence angle
        theta_left = 0.0
        theta_right = np.pi
        theta_mid=None
        print("Searching for the root for theoretical convergence angle")
        while(True):
            theta_mid = (theta_left+theta_right)/2

            #getting the function value at this theta
            func_val = (np.exp(np.cos(theta_mid))+1) / (np.exp(-m * np.cos(theta_mid))+1)
            print("search value:{}\tfunc_value:{}".format(search_val,func_val))
            #Stopping if we found it
            if(np.abs(func_val-search_val)<1e-9):
                break
            
            #Othwise we shift the padestal
            if(func_val>search_val):
                theta_left=theta_mid 
            else:
                theta_right=theta_mid
        
        return theta_mid
    
    def get_point_convergence_theory_bound(self,):
        '''
        '''

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
        self.initialize_topic_related_machinery()
          
    def initialize_topic_related_machinery(self,):
        '''
        This function will used again to create new topic realted layer
        in case we want to retrain the topic classifier for multi-step
        removal.
        '''
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
    
    def train_step(self,data_batch,tidx,task,P_matrix):
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
                train_proj = self._get_proj_X_enc(train_enc,P_matrix)
                train_prob = self.main_task_classifier(train_proj)
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
                train_proj = self._get_proj_X_enc(train_enc,P_matrix)
                train_pred = self.topic_classifier_list[tidx](train_proj)
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
                valid_proj = self._get_proj_X_enc(valid_enc,P_matrix)
                valid_pred = self.topic_classifier_list[tidx](valid_proj)
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
        #Handle the case when P matrix is none
        if type(P_matrix)!=type(np.array([0.1])):
            P_matrix = np.eye(self.args["latent_space_dim"])
        
        #Converting to the numpy array
        P_matrix = tf.constant(P_matrix.T,dtype=tf.float32)

        #Now projecting this latent represention into null space
        X_proj = tf.matmul(X_enc,P_matrix)

        return X_proj 
    
    def get_projected_dataset(self,data_batch,P_matrix):
        '''
        '''
        #Getting the dataset
        all_X = data_batch["X"]
        all_main_Y = data_batch["main_Y"]
        all_topic_Y = data_batch["topic_Y"]

        all_enc = self.pass_via_encoder(all_X)
        X_proj = self._get_proj_X_enc(all_enc,P_matrix)

        return X_proj.numpy(), all_topic_Y.numpy()



def run_convergence_experiment(label_corr_prob,num_rerun):
    convergence_result_dict=defaultdict(list)
    for lprob in label_corr_prob:
        for ridx in range(num_rerun):
            #Getting the required arguments
            args = {}
            args["num_examples"]=1000
            args["batch_size"]=128
            args["valid_split"]=0.2
            args["viz_data"]=False 

            args["lr"]=0.005
            args["main_epochs"]=1
            args["topic_epochs"]=40

            args["num_hlayer"] = 0
            args["hlayer_dim"] = -1


            args["beta"] = 1.0
            args["label_noise"] = 1-lprob  #not-so-perfect correlatedness


            #Buying the 10 rs "remover"
            remover = INLPRemover(args)
            all_topic_angle = remover.trainer_with_inlp(),

            convergence_result_dict[lprob].append(all_topic_angle)
    #Plotting the result
    mean_topic_angle,std_topic_angle =[[],[]],[[],[]]
    print(convergence_result_dict)
    for lprob in label_corr_prob:
        topic1_angle,topic2_angle = zip(*convergence_result_dict[lprob])
        #Getting the mean topic angle across runs
        mean_topic_angle[0].append(np.mean(topic1_angle))
        mean_topic_angle[1].append(np.mean(topic2_angle))
        #Getting the std across runs
        std_topic_angle[0].append(np.std(topic1_angle))
        std_topic_angle[1].append(np.std(topic2_angle))
    
    print(mean_topic_angle,std_topic_angle)
    x=label_corr_prob
    plt.errorbar(x,np.array(mean_topic_angle[0]),fmt="o-",yerr=np.array(std_topic_angle[0]),label="topic1_angle")
    plt.errorbar(x,np.array(mean_topic_angle[1]),fmt="o-",yerr=np.array(std_topic_angle[1]),label="topic2_angle")
    plt.legend()
    plt.ylim((0.0,1.0))
    plt.xlabel("p (perfect-correlatedness)")
    plt.ylabel("angle (multiple of pi)")
    plt.grid()
    plt.show()

def run_removal_experiment(lprob):
    args = {}
    args["num_examples"]=1000
    args["batch_size"]=128
    args["valid_split"]=0.2
    args["viz_data"]=True 

    args["lr"]=0.005
    args["main_epochs"]=1
    args["topic_epochs"]=100

    args["num_hlayer"] = 0
    args["hlayer_dim"] = -1


    args["beta"] = 1.0
    args["label_noise"] = 1-lprob  #not-so-perfect correlatedness

    args["num_inlp_iter"] = 2
    args["remove_tidx"] = 0

    remover = INLPRemover(args)
    remover.trainer_with_inlp()

def run_convergence_with_noise_experiment(noise_mean,noise_sigma,num_rerun):
    '''
    '''
    fig,ax = plt.subplots(len(noise_sigma),2,gridspec_kw={'width_ratios': [3, 1]})
    for sidx,sigma in enumerate(noise_sigma):
        convergence_angle=defaultdict(list)
        theory_converged_angle=defaultdict(list)
        topic1_accuracy = defaultdict(list)
        for noise in noise_mean:
            for rnum in range(num_rerun):
                args = {}
                args["num_examples"]=1000
                args["batch_size"]=128
                args["valid_split"]=0.2
                args["viz_data"]=False

                args["lr"]=0.005
                args["main_epochs"]=1
                args["topic_epochs"]=40

                args["num_hlayer"] = 0
                args["hlayer_dim"] = -1


                args["beta"] = 1.0
                args["label_noise"] = 0.5  #not-so-perfect correlatedness

                args["inject_noise"]=True
                args["noise_mean"] = noise
                args["noise_sigma"] = sigma


                #Buying the 10 rs "remover"
                remover = INLPRemover(args)
                all_topic_angle,topic1_acc = remover.trainer_with_inlp()
                theory_angle = remover.get_converged_angle_with_noise_theory()/np.pi

                #Saving the converged angle
                convergence_angle[noise].append(all_topic_angle[0])
                theory_converged_angle[noise].append(theory_angle)
                topic1_accuracy[noise].append(topic1_acc)
    
        mean_angle,std_angle = [],[]
        mean_acc, std_acc = [],[]
        mean_cangle,std_cangle = [],[]
        #Plotting the result
        for noise in noise_mean:
            mean_angle.append(np.mean(convergence_angle[noise]))
            std_angle.append(np.std(convergence_angle[noise]))

            mean_acc.append(np.mean(topic1_accuracy[noise]))
            std_acc.append(np.std(topic1_accuracy[noise]))

            mean_cangle.append(np.mean(theory_converged_angle[noise]))
            std_cangle.append(np.std(theory_converged_angle[noise]))
        
        ax[sidx,0].errorbar(noise_mean,mean_angle,yerr=std_angle,fmt="-o",label="converged (sigma={:0.3f})".format(sigma))
        ax[sidx,0].errorbar(noise_mean,mean_cangle,yerr=std_cangle,fmt="-o",label="theory (sigma={:0.3f})".format(sigma))
        ax[sidx,1].errorbar(noise_mean,mean_acc,yerr=std_acc,fmt="-o")#,label="sigma={}".format(sigma))

    
    
        ax[sidx,0].set_ylim((0.0,1.0))
        ax[sidx,0].set_xlabel("noise mean")
        ax[sidx,0].set_ylabel("angle (multiple of pi)")
        ax[sidx,0].legend()
        ax[sidx,0].grid(True)

        ax[sidx,1].set_ylim((0.0,1.0))
        ax[sidx,1].set_xlabel("noise mean")
        ax[sidx,1].set_ylabel("topic accuracy")
        ax[sidx,1].legend()
        ax[sidx,1].grid(True)
    plt.show()

def run_convergece_with_noise_with_theory(mvals,epsilon,num_alpha):
    fig,ax = plt.subplots(len(mvals),2,gridspec_kw={'width_ratios': [3, 1]})
    for midx,m in enumerate(mvals):
        #Getting the bound
        LB = (1+np.exp(-1))/(1+np.exp(m))
        UB = (1+np.exp(1))/(1+np.exp(-m))

        #Now creating the alpha range based on the mvalue 
        LB_alpha = (LB - epsilon)*m
        UB_alpha = (UB + epsilon)*m

        LB_alpha = LB_alpha if LB_alpha>1.0 else 1.0

        print("\n\n\n\nm:{}\tLB_alpha:{}\tUB_alpha:{}".format(m,LB_alpha,UB_alpha))

        alpha = np.linspace(LB_alpha,UB_alpha,num=num_alpha)
        noise_ratios = (1.0/(1+alpha)).tolist()#this is alpha_minor
        print("alpha_list:",noise_ratios)

        alpha_by_m = (alpha/m).tolist()

        expected_angle = []
        actual_angle = []
        valid_accuracy = []
        for nratio in noise_ratios:
            args = {}
            args["num_examples"]=1000
            args["batch_size"]=128
            args["valid_split"]=0.2
            args["viz_data"]=False

            args["lr"]=0.005
            args["main_epochs"]=1
            args["topic_epochs"]=40

            args["num_hlayer"] = 0
            args["hlayer_dim"] = -1

            #This is for second topic so not relavant here
            args["beta"] = 1.0
            args["label_noise"] = 0.5  #not-so-perfect correlatedness


            args["inject_noise"] = True
            args["noise_ratio"]=nratio #amount of noise --> alpha value
            args["noise_pos"] = m #the relative noise location --> m value


            #Buying the 10 rs "remover"
            remover = INLPRemover(args)
            all_topic_angle,topic1_acc = remover.trainer_convergence()
            # all_topic_angle,topic1_acc = remover.trainer_with_inlp()
            theory_converged_angle = remover.get_converged_angle_with_noise_theory()/np.pi

            print("converged angle: theory:{}\tactual:{}".format(
                                                theory_converged_angle,
                                                all_topic_angle[0])
            )

            #Adding the angle to the list
            expected_angle.append(theory_converged_angle)
            actual_angle.append(all_topic_angle[0])
            valid_accuracy.append(topic1_acc)
        
        #Plotting the result
        ax[midx,0].plot(alpha_by_m,expected_angle,"-o",label="expected")
        ax[midx,0].plot(alpha_by_m,actual_angle,"-o",label="converged")

        #Plotting the upper and lowe bound
        num_yline_point = 10
        y_line = np.linspace(0.0,1.0,num_yline_point)
        LB_x = [LB,]*num_yline_point
        UB_x = [UB,]*num_yline_point

        ax[midx,0].plot(LB_x,y_line,"--",c='k')
        ax[midx,0].plot(UB_x,y_line,"--",c='k')


        ax[midx,0].grid(True)
        ax[midx,0].set_ylim((0.0,1.0))
        ax[midx,0].set_xlabel("alpha/m (m={})".format(m))
        ax[midx,0].set_ylabel("angle (multiple of pi)")
        ax[midx,0].legend()
        # ax[midx].title.set_text('noise_position (m) = {}'.format(m))

        ax[midx,1].plot(alpha_by_m,valid_accuracy,"-o")
        ax[midx,1].set_ylim((0.0,1.1))
        ax[midx,1].grid(True)
        ax[midx,1].set_xlabel("alpha/m (m={})".format(m))
        ax[midx,1].set_ylabel("Valid Acc")
    
    plt.show()

def test_func1():
    '''
    Just testing how does the plot of the convergence equation varies as we
    change the value of theta.
    '''
    all_m= np.linspace(0.1,1.0,3).tolist()+ [2.0,4.0,8.0,16.0] #np.linspace(0.1,128,num=10).tolist()
    theta = np.linspace(0.0,np.pi,num=50)

    sol_theta = []
    for m in all_m:
        # func_val = (np.exp(np.cos(theta))+1)/(np.exp(-m*np.cos(theta))+1)
        # plt.plot(theta,func_val,label="m={:0.3f}".format(m))
        # plt.plot(theta,theta*0.0+(1.0/m),label="1/m")
    
        # plt.legend()
        # plt.show()
        search_val = 1.0/m
        print("\n\n\nFinding the sol for m: {}\t search_val:{}".format(m,search_val))

        
        UB = (1+np.exp(1))/(1+np.exp(-m))
        if(search_val>UB):
            sol_theta.append(0.0)
            continue
        #there wont be a violation of the upper bound

        #Getting the solution for this m in theta
        theta_left = 0.0
        theta_right = np.pi 
        while(True):
            theta_mid = (theta_left + theta_right)/2
            func_val = (np.exp(np.cos(theta_mid))+1)/(np.exp(-m*np.cos(theta_mid))+1)
            print("search_val:{}\tfunc_val:{}".format(search_val,func_val))

            if(np.abs(func_val-search_val)<1e-8):
                sol_theta.append(theta_mid)
                break 
            
            #Otherwise we will move our position
            if(func_val>search_val):
                theta_left = theta_mid 
            else:
                theta_right = theta_mid
            
    
    plt.plot(all_m,sol_theta,"-o")
    plt.show()

    # m = np.linspace(0.0,2.0,num=100)
    # plt.plot(np.exp(m-1))
    # plt.plot(m)
    # plt.show()



if __name__=="__main__":
    #CONVERGENCE EXPERIMENT
    label_corr_prob = np.arange(0.5,1.0,0.1)
    num_rerun = 5
    # run_convergence_experiment(label_corr_prob,num_rerun)

    #REMOVAL and new TOPIC direction Experiment
    # run_removal_experiment(0.5)

    #Convergence with noise experiment
    noise_mean = [0.3,0.4,0.5,0.6] #np.arange(0.7,1.5,0.1).tolist()
    noise_sigma = np.arange(0.1,0.26,0.05).tolist()
    num_rerun=1
    run_convergence_with_noise_experiment(noise_mean,noise_sigma,num_rerun)

    # Convergence with noise experiment with theory
    # mvals = [0.5,1.0,2.0,3.0]                    #noise position
    # run_convergece_with_noise_with_theory(
    #                                 mvals=mvals,
    #                                 epsilon=1.0,
    #                                 num_alpha=5,
    # )

    # test_func1()
    


    
    


