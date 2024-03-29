import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import pdb
from pprint import pprint
np.random.seed(14)

import matplotlib.pyplot as plt

class Synthetic1():
    '''
    This class will contain the synthetic tabular dataset generator and
    its corresponding trained predictor.
    '''
    def __init__(self,args):
        self.num_examples=None#args["num_examples"]
    
    def generate_dataset(self,num_examples,interv_val):
        '''
        '''
        x_y = -10+ np.random.randn(num_examples)         #stable
        x_p = np.random.randn(num_examples)         #unstable features

        if interv_val==None:
            x_p =  x_y + x_p
        else:
            x_p = interv_val + x_p
        
        #Aggregating the X
        X = np.stack([x_y,x_p],axis=1)
        
        #Creating the labels
        Y = x_y>=-10#np.mean(x_y)

        assert X.shape[0]==Y.shape[0]

        return (X,Y)
    
    def get_predictor(self,X,Y,valid_split=0.2):
        '''
        '''
        #Creting the predictor
        x = keras.Input(shape=(2,),dtype="float64")
        # h = layers.Dense(2, activation=tf.keras.layers.LeakyReLU(alpha=0.0))(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(x, output)
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(x=X,y=Y, epochs=100, validation_split=valid_split)

        self.predictor=model
    
    def remove_spurious_features(self,X1,Y1,X2,Y2):
        '''
        '''
        print("#################################################")
        print("###########  Training the Debugger ##############")
        print("#################################################")
        #Creating the discriminator
        discriminator = keras.Sequential(
            [
                layers.InputLayer((2,)),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        self.discriminator = discriminator
        

        #Creating the generator
        generator = keras.Sequential(
            [
                layers.InputLayer((2,)),
                layers.Dense(2),
                layers.LeakyReLU(alpha=0.2),
            ],
            name="generator",
        )
        self.generator = generator

        #Now creting the Debugger model
        debugger = Debugger(generator,discriminator,self.predictor)
        debugger.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
            p_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
        )


        #Creting the actual domain dataset 
        # X = np.concatenate([X1,X2],axis=0)
        # Y = np.concatenate([np.zeros(X1.shape[0]),np.ones(X2.shape[0])],axis=0)
        d1_dataset = tf.data.Dataset.from_tensor_slices((X1,Y1))
        d1_dataset = d1_dataset.shuffle(buffer_size=1024)

        d2_dataset = tf.data.Dataset.from_tensor_slices((X2,Y2))
        d2_dataset = d2_dataset.shuffle(buffer_size=1024)

        #Fitting the debugger first
        dataset = tf.data.Dataset.zip((d1_dataset,d2_dataset))
        dataset = dataset.shuffle(buffer_size=2048).batch(128)

        debugger.fit(dataset,epochs=200)


        #Now using the debugger to remove the spurious features
        stable_X1 = self.generator(X1)
        #Also let see how much the predictor is surprised
        stable_Y1 = tf.argmax(self.predictor(stable_X1),axis=1).numpy()
        pprint(stable_Y1==Y1)
        print("Accuracy:",np.mean(stable_Y1==Y1))

        

        #Now lets see how good we are able to remove the spurious features
        plt.hist(X1[:,0],edgecolor='k',alpha=0.5,label="train-domain1")
        plt.hist(X2[:,0],edgecolor='k',alpha=0.5,label="domain2")
        plt.hist(stable_X1[:,0],edgecolor='k',alpha=0.5,label="stablized")
        plt.legend()
        plt.show()


        plt.hist(X1[:,1],edgecolor='k',alpha=0.5,label="train-domain1")
        plt.hist(X2[:,1],edgecolor='k',alpha=0.5,label="domain2")
        plt.hist(stable_X1[:,1],edgecolor='k',alpha=0.5,label="stablized")
        plt.legend()
        plt.show()


        #Seeing all of them in one scatter plot
        plt.scatter(X1[:,0],X1[:,1],label="train-domain1")
        plt.scatter(X2[:,0],X2[:,1],label="domain2")
        plt.scatter(stable_X1[:,0],stable_X1[:,1],label="stablized")
        plt.legend()
        plt.xlabel("stable_dimension")
        plt.ylabel("spurious_dimension")
        plt.show()


class MNISTTransform():
    '''
    This class will generate the MNIST dataset from different domain, where
    different domains contains different transformation on the inout images.
    '''
    def __init__(self,args):
        self.args=args
    
    def generate_dataset(self,num_examples,class_list,transformation,angle):
        (X_all,Y_all),_ = tf.keras.datasets.mnist.load_data()
        print("Total Number of examples in dataset: {}".format(X_all.shape[0]))


        #Now we will only retreive the class which is needed
        X, Y = [], []
        class2label_dict = {}
        for lidx,class_label in enumerate(class_list):
            #relabelling the images-class for logits
            class2label_dict[class_label] = lidx 

            #Filtering the images
            filter_arr = (Y_all == class_label)
            X_class = X_all[filter_arr][0:num_examples]
            Y_class = np.ones(X_class.shape[0])*lidx 

            X.append(X_class)
            Y.append(Y_class)
        #Concatenating all the images
        X = np.concatenate(X,axis=0)
        Y = np.concatenate(Y,axis=0)

        #Now filtering the whole array
        X = np.expand_dims(X,axis=-1).astype("float32")/255.0
        print("Number of Example after Subset: {}".format(X.shape))

        #Now creating the domain2 examples
        if transformation=="rotation":
            X = tfa.image.rotate(X,[angle]*X.shape[0]).numpy()
        
        # pdb.set_trace()
        return X,Y
    
    def get_predictor(self,X,Y,class_list,valid_split=0.2):
        '''
        '''
        predictor = keras.Sequential(
            [
                layers.InputLayer((28,28,1)),

                layers.Conv2D(32, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),

                layers.Flatten(),
                layers.Dense(64),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(len(class_list),activation="softmax")
            ],
            name="predictor",
        )
        self.predictor = predictor
        print("###########################################")
        print("#########  TRAINING THE PREDICTOR  ########")
        print("###########################################")
        print(predictor.summary())

        #Training the model
        predictor.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        predictor.fit(
                X,Y,epochs=5,validation_split=valid_split,
        )
    
    def remove_spurious_features(self,X1,Y1,X2,Y2,class_list,valid_split=0.2):
        '''
        '''
        print("#################################################")
        print("###########  Training the Debugger ##############")
        print("#################################################")
        #Creating the discriminator
        discriminator = keras.Sequential(
            [
                layers.InputLayer((28,28,1)),

                layers.Conv2D(32, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),

                layers.Flatten(),
                layers.Dense(64),
                layers.LeakyReLU(alpha=0.2),
                
                layers.Dense(1,activation="sigmoid")
            ],
            name="discriminator",
        )
        self.discriminator = discriminator
        print("###########################################")
        print("#######  TRAINING THE DISCRIMINATOR  ######")
        print("###########################################")
        print(discriminator.summary())

        #Training the discriminator first
        discriminator.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        #Creating the dataset for discriminator
        X_disc = np.concatenate([X1,X2],axis=0)
        Y_disc = np.concatenate([np.zeros(X1.shape[0]),np.ones(X2.shape[0])],axis=0)

        discriminator.fit(
                X_disc,Y_disc,epochs=5,validation_split=valid_split,
        )
        

        #Creating the generator
        gen_compressed_dim=1
        generator = keras.Sequential(
            [
                layers.InputLayer((28,28,1)),

                layers.Conv2D(32, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.2),

                layers.Flatten(),
                layers.Dense(7*7*gen_compressed_dim),
                layers.LeakyReLU(alpha=0.2),



                layers.Reshape((7, 7, gen_compressed_dim)),
                layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),


                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),

            ],
            name="generator",
        )
        self.generator = generator
        print("###########################################")
        print("########  TRAINING THE DEBUGGER  #########")
        print("###########################################")
        print(generator.summary())

        debugger = Debugger(generator,discriminator,self.predictor)
        debugger.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
            p_optimizer=keras.optimizers.Adam(learning_rate=0.0009),
        )


        #Creting the actual domain dataset 
        # X = np.concatenate([X1,X2],axis=0)
        # Y = np.concatenate([np.zeros(X1.shape[0]),np.ones(X2.shape[0])],axis=0)
        d1_dataset = tf.data.Dataset.from_tensor_slices((X1,Y1))
        d1_dataset = d1_dataset.shuffle(buffer_size=1024)

        d2_dataset = tf.data.Dataset.from_tensor_slices((X2,Y2))
        d2_dataset = d2_dataset.shuffle(buffer_size=1024)

        #Fitting the debugger first
        dataset = tf.data.Dataset.zip((d1_dataset,d2_dataset))
        dataset = dataset.shuffle(buffer_size=2048).batch(64)

        debugger.fit(dataset,epochs=10)

        #Plotting some sample example
        stable_X1 = self.generator(X1).numpy()
        
        
        figure, axes = plt.subplots(2,2)

        idx0 = 123
        axes[0,0].imshow(X1[idx0,:,:,0],cmap="gray")
        axes[0,1].imshow(stable_X1[idx0,:,:,0],cmap="gray")

        idx1 = 6564
        axes[1,0].imshow(X1[idx1,:,:,0],cmap="gray")
        axes[1,1].imshow(stable_X1[idx1,:,:,0],cmap="gray")

        plt.show()
        
        

        pdb.set_trace()


class Debugger(keras.Model):
    '''
    '''
    def __init__(self,generator,discriminator,predictor):
        '''
        '''
        super(Debugger,self).__init__()

        #Assignign the main layers
        self.generator = generator 
        self.discriminator = discriminator
        self.predictor = predictor

        #Assigning the debugging metrics
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.pred_loss_tracker = keras.metrics.Mean(name="predictor_loss")

        #Individual debuggin metrics
        self.gen_pred_loss = keras.metrics.Mean(name="gen_pred_loss")
        self.gen_disc_loss = keras.metrics.Mean(name="gen_disc_loss")
        self.gen_sem_loss  = keras.metrics.Mean(name="gen_sem_loss")
        # self.gen_norm_loss = keras.metrics.Mean(name="gen_norm_loss")

        # self.disc_stable_loss = keras.metrics.Mean(name="disc_stable_loss")
        self.disc_actual_loss = keras.metrics.Mean(name="disc_actual_loss")
        self.disc_actual_acc  = keras.metrics.BinaryAccuracy(name="disc_stable_acc",threshold=0.5)


        self.pred_stable_loss = keras.metrics.Mean(name="pred_stable_loss")
        self.pred_actual_loss = keras.metrics.Mean(name="pred_actual_loss")
        self.pred_stable_acc  = keras.metrics.SparseCategoricalAccuracy(name="pred_stable_acc")
        self.pred_actual_acc  = keras.metrics.SparseCategoricalAccuracy(name="pred_actual_acc")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, p_optimizer):
        super(Debugger, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.p_optimizer = p_optimizer
        #self.loss_fn = loss_fn
    
    def train_step(self,data):
        '''
        '''
        (X1,Y1),(X2,Y2) = data
        # pdb.set_trace()
        #Concatenating the data for generator
        X = tf.concat([X1,X2],axis=0)
        Y = tf.concat(
            [
                tf.zeros(tf.shape(Y1)[0]),
                tf.ones(tf.shape(Y2)[0])
            ],
            axis=0,
        )
        #Generating the noise removed data from generator
        stable_X1 = self.generator(X1)

        #Defining the loss function
        bxentropy_loss = keras.losses.BinaryCrossentropy(from_logits=False)
        scxentropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        mse = keras.losses.MeanSquaredError()

        def get_pred_entropy_loss(logits):
            pred_entropy = -1*tf.math.reduce_sum(
                                        logits*tf.math.log(logits),
                                        axis=1,
            )
            entropy_loss = -1* tf.math.reduce_mean(pred_entropy,axis=0)

            return entropy_loss

        #Training the discriminator
        with tf.GradientTape() as tape:
            #These should be considerend as high entorpy prediction
            # stable_pred = self.discriminator(stable_X1)
            # #We want to maximize the entropy of prediction
            # stable_loss = get_pred_entropy_loss(stable_pred)

            #For these the prediciton should be correct domain
            actual_pred = self.discriminator(X)
            actual_loss = bxentropy_loss(Y,actual_pred)

            #Calculating the total loss
            disc_loss = actual_loss #+ stable_loss
        
        #Now we will optimize the discriminator
        grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        #Updating all the lossses for discriminator
        self.disc_loss_tracker.update_state(disc_loss)
        self.disc_actual_loss.update_state(actual_loss)
        self.disc_actual_acc.update_state(Y,actual_pred)
        # self.disc_stable_loss.update_state(stable_loss)





        #Now we want to train the generator
        with tf.GradientTape() as tape:
            #Generating the stablized data with noise removed
            stable_X1 = self.generator(X1)

            #Constraint 1: Prediction should remain same
            predictor_stable_pred = self.predictor(stable_X1)
            predictor_actual_pred = tf.math.argmax(self.predictor(X1),axis=1)
            gen_predictor_loss = scxentropy_loss(Y1,
                                            predictor_stable_pred
                )
            #Constarint 1a: keep them semantically similar
            semantic_loss = mse(X1,stable_X1)

            # #Adding the norm loss, to encourage deletion
            # norm_loss = mse(0,stable_X1)

            
            #Constraint 2: High Entropy for Discriminator
            disc_stable_pred = self.discriminator(stable_X1)
            gen_disc_loss = get_pred_entropy_loss(disc_stable_pred)

            #Getting the total generator loss
            gen_loss = 5*gen_disc_loss + 5*gen_predictor_loss + semantic_loss #+ 0.01*norm_loss  #+semantic_loss #+10*gen_predictor_loss
        
        #Now optimizing the generator
        grads = tape.gradient(gen_loss,self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        #Updating the losssses for generator
        self.gen_loss_tracker.update_state(gen_loss)
        self.gen_pred_loss.update_state(gen_predictor_loss)
        self.gen_disc_loss.update_state(gen_disc_loss)
        self.gen_sem_loss.update_state(semantic_loss)
        # self.gen_norm_loss.update_state(norm_loss)





        #Training the predictor also to update based on generated samples
        with tf.GradientTape() as tape:
            #Getting the predicotr loss on the actual inputs
            pred_actual_logits = self.predictor(X1)
            pred_actual_loss = scxentropy_loss(Y1,pred_actual_logits)

            #Getting the predictor loss on stablized inputs
            stable_X1 = self.generator(X1)
            pred_stable_logits = self.predictor(stable_X1)
            pred_stable_loss = scxentropy_loss(Y1,pred_stable_logits)

            pred_loss = pred_actual_loss+pred_stable_loss
        #Now optimizing the generator
        grads = tape.gradient(pred_loss,self.predictor.trainable_weights)
        self.p_optimizer.apply_gradients(
            zip(grads,self.predictor.trainable_weights)
        )
        self.pred_loss_tracker.update_state(pred_loss)
        self.pred_actual_loss.update_state(pred_actual_loss)
        self.pred_stable_loss.update_state(pred_stable_loss)
        self.pred_actual_acc.update_state(Y1,pred_actual_logits)
        self.pred_stable_acc.update_state(Y1,pred_stable_logits)



        
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "p_loss": self.pred_stable_loss.result(),
            "g_p_loss": self.gen_pred_loss.result(),
            "g_d_loss": self.gen_disc_loss.result(),
            "g_s_loss": self.gen_sem_loss.result(),
            #"g_n_loss": self.gen_norm_loss.result(),
            #"d_s_loss": self.disc_stable_loss.result(),
            #"d_a_loss": self.disc_actual_loss.result(),
            "d_a_acc": self.disc_actual_acc.result(),
            #"p_a_loss":self.pred_actual_loss.result(),
            #"p_s_loss":self.pred_stable_loss.result(),
            "p_a_acc":self.pred_actual_acc.result(),
            "p_s_acc":self.pred_stable_acc.result()
        }


if __name__=="__main__":
    # predictor = Synthetic1(args=None)
    # X1,Y1 = predictor.generate_dataset(num_examples=10000,
    #                                 interv_val=None,
    #                     )
    # predictor.get_predictor(X=X1,Y=Y1,valid_split=0.2)

    # #Getting the dataset from other domain
    # X2,Y2 = predictor.generate_dataset(num_examples=10000,
    #                                    interv_val=10.0,
    #                     )
    # predictor.remove_spurious_features(X1,Y1,X2,Y2)
    

    predictor = MNISTTransform(args=None)
    class_list = [1,2]  #for large number of class we need more complex model
    X1,Y1 = predictor.generate_dataset(num_examples=5000,
                                class_list=class_list,
                                transformation="rotation",
                                angle = np.pi/2,
    )
    predictor.get_predictor(X1,Y1,class_list)

    #Now getting the data from other domain
    X2,Y2 = predictor.generate_dataset(num_examples=5000,
                                class_list=class_list,
                                transformation="rotation",
                                angle = 0,
    )
    #Now training the debugger
    predictor.remove_spurious_features(X1,Y1,X2,Y2,class_list)


    pdb.set_trace()





# stable_X1 = self.generator(X1).numpy()
        
        
# figure, axes = plt.subplots(2,2)

# idx0 = 1235
# axes[0,0].imshow(X1[idx0,:,:,0],cmap="gray")
# axes[0,1].imshow(stable_X1[idx0,:,:,0],cmap="gray")

# idx1 = 8564
# axes[1,0].imshow(X1[idx1,:,:,0],cmap="gray")
# axes[1,1].imshow(stable_X1[idx1,:,:,0],cmap="gray")

# plt.show()

