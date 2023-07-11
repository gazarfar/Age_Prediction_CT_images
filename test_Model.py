# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:15:17 2023

@author: azarf
"""

import tensorflow as tf
import numpy as np




feature_data ={'NormalizeCompositeImage': tf.io.FixedLenFeature([256,256,3], tf.float32),
               'label': tf.io.FixedLenFeature([], tf.float32),
               'ID': tf.io.FixedLenFeature([], tf.int64)}


 
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_data)

def normalize(input):
    case = input['NormalizeCompositeImage']
    label = input['label']
    #  ID = input['ID']
    case = tf.where(tf.math.is_nan(case), 0., case)
    mean, variance = tf.nn.moments(case, axes=[0,1])
    x_normed = (case - mean) / tf.sqrt(variance + 1e-8) # epsilon to avoid dividing by zero
    x_normed = tf.transpose(x_normed, perm=[2,0,1])
    y = tf.map_fn(fn=lambda chanel: tf.divide(chanel-tf.reduce_min(chanel),tf.reduce_max(chanel)-tf.reduce_min(chanel)), elems=x_normed, fn_output_signature=tf.float32)
    y = tf.transpose(y, perm=[1,2,0])
    return (y,label)

def read_file(filename):
    raw_dataset  = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.shuffle(len(filename))
    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.map(normalize)
    return parsed_dataset


def test(dataset,model):
    data_size = sum(1 for _ in dataset)
    original = []
    for images, label in dataset.take(data_size):
        original = np.append(original,label.numpy())
    original = np.reshape(original,[original.shape[0],1])
    predicted = model.predict(dataset)
    original= np.ndarray.round(original*25+52.5)
    predicted= predicted*25+52.5
    offset = -0.988*original+62.896
    predicted = np.ndarray.round(predicted - offset)
    return original, predicted

path2tfrecords = 'C:\\Users\\azarf\\Documents\\Age_prediction_Spyder\\Tfrecords_path\\'
tfrecordname = 'CompositeImage.tfrecords'
path2model = 'C:\\Users\\azarf\\Documents\\AgePrediction_20230503\\my_model_fine2\\my_model_fine2\\'

dataset = read_file(path2tfrecords + tfrecordname)
dataset = dataset.batch(40)

model = tf.keras.models.load_model(path2model)

Chronological_age, Estimeted_age = test(dataset,model)