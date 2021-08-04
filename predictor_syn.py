import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pdb
from pprint import pprint

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
        x_y = -10 + np.random.randn(num_examples)         #stable
        x_p = np.random.randn(num_examples)         #unstable features

        if interv_val==None:
            x_p =  x_y + x_p
        else:
            x_p = interv_val + x_p
        
        #Aggregating the X
        X = np.stack([x_y,x_p],axis=1)
        
        #Creating the labels
        Y = x_y>=0

        assert X.shape[0]==Y.shape[0]

        return (X,Y)
    
    def get_predictor(self,X,Y,valid_split=0.2):
        '''
        '''
        #Creting the predictor
        x = keras.Input(shape=(2,),dtype="float64")
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(x, output)
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(x=X,y=Y, epochs=50, validation_split=0.2)

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
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        )


        #Creting the actual domain dataset 
        X = np.concatenate([X1,X2],axis=0)
        Y = np.concatenate([np.zeros(X1.shape[0]),np.ones(X2.shape[0])],axis=0)
        debugger.fit(x=X,y=Y,epochs=50)


        #Now using the debugger to remove the spurious features
        stable_X1 = self.generator(X1)
        #Also let see how much the predictor is surprised
        stable_Y1 = tf.argmax(self.predictor(stable_X1),axis=1).numpy()
        pprint(stable_Y1==Y1)
        print("Accuracy:",np.mean(stable_Y1==Y1))

        

        #Now lets see how good we are able to remove the spurious features
        plt.hist(X1[:,0],edgecolor='k',alpha=0.5)
        plt.hist(X2[:,0],edgecolor='k',alpha=0.5)
        plt.hist(stable_X1[:,0],edgecolor='k',alpha=0.5)
        plt.show()


        plt.hist(X1[:,1],edgecolor='k',alpha=0.5)
        plt.hist(X2[:,1],edgecolor='k',alpha=0.5)
        plt.hist(stable_X1[:,1],edgecolor='k',alpha=0.5)
        plt.show()





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

        #Individual debuggin metrics
        self.gen_pred_loss = keras.metrics.Mean(name="gen_pred_loss")
        self.gen_disc_loss = keras.metrics.Mean(name="gen_disc_loss")

        self.disc_stable_loss = keras.metrics.Mean(name="disc_stable_loss")
        self.disc_actual_loss = keras.metrics.Mean(name="disc_actual_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer):
        super(Debugger, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        #self.loss_fn = loss_fn
    
    def train_step(self,data):
        '''
        '''
        X,Y = data
        #Generating the noise removed data from generator
        stable_X = self.generator(X)

        #Defining the loss function
        bxentropy_loss = keras.losses.BinaryCrossentropy(from_logits=True)

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
            stable_pred = self.discriminator(stable_X)
            #We want to maximize the entropy of prediction
            stable_loss = get_pred_entropy_loss(stable_pred)

            #For these the prediciton should be correct domain
            actual_pred = self.discriminator(X)
            actual_loss = bxentropy_loss(Y,actual_pred)

            #Calculating the total loss
            disc_loss = actual_loss + stable_loss
        
        #Now we will optimize the discriminator
        grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        #Updating all the lossses for discriminator
        self.disc_loss_tracker.update_state(disc_loss)
        self.disc_actual_loss.update_state(actual_loss)
        self.disc_stable_loss.update_state(stable_loss)




        #Now we want to train the generator
        with tf.GradientTape() as tape:
            #Generating the stablized data with noise removed
            stable_X = self.generator(X)

            #Constraint 1: Prediction should remain same
            predictor_stable_pred = self.predictor(stable_X)
            predictor_actual_pred = tf.math.argmax(self.predictor(X),axis=1)
            gen_predictor_loss = bxentropy_loss(predictor_actual_pred,
                                            predictor_stable_pred
                )
            
            #Constraint 2: High Entropy for Discriminator
            disc_stable_pred = self.discriminator(stable_X)
            gen_disc_loss = get_pred_entropy_loss(disc_stable_pred)

            #Getting the total generator loss
            gen_loss = gen_predictor_loss + gen_disc_loss
        
        #Now optimizing the generator
        grads = tape.gradient(gen_loss,self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        #Updating the losssses for generator
        self.gen_loss_tracker.update_state(gen_loss)
        self.gen_pred_loss.update_state(gen_predictor_loss)
        self.gen_disc_loss.update_state(gen_disc_loss)
        
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "g_p_loss": self.gen_pred_loss.result(),
            "g_d_loss": self.gen_disc_loss.result(),
            "d_s_loss": self.disc_stable_loss.result(),
            "d_a_loss": self.disc_actual_loss.result()
        }


if __name__=="__main__":
    predictor = Synthetic1(args=None)
    X1,Y1 = predictor.generate_dataset(num_examples=1000,
                                    interv_val=None,
                        )
    predictor.get_predictor(X=X1,Y=Y1,valid_split=0.2)

    #Getting the dataset from other domain
    X2,Y2 = predictor.generate_dataset(num_examples=1000,
                                       interv_val=10.0,
                        )
    predictor.remove_spurious_features(X1,Y1,X2,Y2)

    # pdb.set_trace()








