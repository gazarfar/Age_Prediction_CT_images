# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:56:09 2023

@author: azarf
"""
import tensorflow as tf

def My_ResNet(input_shape):
   


    base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,weights='imagenet',
                                                                             input_shape=input_shape)
     # freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    x = base_model(inputs, training=False) 
    
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
        
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    ### END CODE HERE
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

## Training the model

model = My_ResNet((256,256,3),(256,256,3))



initial_learning_rate = 0.0001
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
 #   initial_learning_rate,
 #   decay_steps=10,
 #   decay_rate=0.1,
  #  staircase=True) 


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae','mse'])
model.summary()

