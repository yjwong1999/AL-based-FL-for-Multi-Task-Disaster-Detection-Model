# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:54:37 2022

@author: e-default
"""
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import List

import numpy as np
import pandas as pd 
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logger = logging.getLogger(__name__)

"""Image Paths and Labels"""
print('Getting Data')
data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
annot_train_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv')
annot_dev_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv')
annot_test_path = os.path.join(data_root, 'tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv')
    
# training data
df = pd.read_csv(annot_train_path, sep='\t')
train_img_paths = np.array(df['image_path'].tolist())
train_labels = np.array(df['class_label'].tolist())

# dev data
df = pd.read_csv(annot_dev_path, sep='\t')
dev_img_paths = np.array(df['image_path'].tolist())
dev_labels = np.array(df['class_label'].tolist())

# test data
df = pd.read_csv(annot_test_path, sep='\t')
test_img_paths = np.array(df['image_path'].tolist())
test_labels = np.array(df['class_label'].tolist())

# get all unique labels
unique_label = np.unique(train_labels)

# convert data root to tf string
data_root = tf.convert_to_tensor(data_root, tf.string)


"""model defintition"""
import tensorflow as tf
from models import get_disaster_head

# get augmentor
def get_augmentor():
    augmentor = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, 0.1),
    ])
    return augmentor
  
# get the backbone
def get_backbone(size, class_num, backbone_h5):
    # define input
    in_shape = (size, size, 3)
    input_images = tf.keras.layers.Input(shape=in_shape)
    x = input_images
    # load the backbone
    backbone = tf.keras.models.load_model(backbone_h5)
    backbone.trainable = False
    _, x, _ = backbone(x)
    # define the model
    model = tf.keras.models.Model(input_images, x, name='backbone')
    return model

  
# get head model
def create_model():
    x = np.ones((1, 26, 26, 512))
    x = tf.convert_to_tensor(x)
    disaster_head = get_disaster_head(x, class_num=7)
    return disaster_head

  
backbone = get_backbone(size=416, class_num=7, backbone_h5='backbone.h5')
disaster_predictor = create_model()
augmentor = get_augmentor()

"""Training Utilities"""
class TrainingUtils:
    def __init__(self, data_root, unique_label):
        self.data_root = tf.convert_to_tensor(data_root, tf.string)
        self.unique_label = unique_label
    
    def load_data(self, data):
        # split the data into x and y (image and label)
        image_path = data[0]
        label = data[1]
        # read the image from disk, decode it, resize it, and scale the
        # pixels intensities to the range [0, 1]
        image_path = tf.strings.join([self.data_root, image_path])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416)) / 255.0
        # encode the label
        label = tf.argmax(label == self.unique_label)
        # return the image and the integer encoded label
        return (image, label)
    
      
    @tf.function
    def feature_extraction_augment(self, image, label):
        # perform random horizontal and vertical flips
        image = tf.image.random_flip_left_right(image)
        # brightness
        image = tf.image.random_brightness(image, 0.2)
        # contrast
        image = tf.image.random_contrast(image, 0.5, 2.0)
        # saturation
        image = tf.image.random_saturation(image, 0.80, 1.20) #ori is 0.75-1.25
        # rotation/translation/zoom
        image = tf.expand_dims(image, axis=0)
        image = augmentor(image)
        # feature extraction with augmentation
        tensor = backbone(image, training=False)[0]
        # return the tensor and the label
        return (tensor, label)
     
      
    @tf.function
    def feature_extraction(self, image, label):
        # feature extraction
        image = tf.expand_dims(image, axis=0)
        tensor = backbone(image, training=False)[0]
        # return the tensor and the label
        return (tensor, label)
    
      
    def onehot(self, tensor, label):
        label = tf.one_hot(label, 7)
        return (tensor, label)
      


"""Create tf dataset"""
data = list(zip(test_img_paths, test_labels))
testDS = tf.data.Dataset.from_tensor_slices(data)
            
utils = TrainingUtils(data_root, unique_label)      

testDS = (testDS
	.map(utils.load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .map(utils.feature_extraction, num_parallel_calls=tf.data.AUTOTUNE)
	.batch(32)
	.prefetch(tf.data.AUTOTUNE)
)


"""Get Encodings and Labels"""
path = 'best_model_al_fl_easy_mod_hard.h5'
path = 'best_model_2_envoy_30_epoch.h5'
disaster_predictor.load_weights(path)
# disaster_predictor.summary()
    
encoder = tf.keras.models.Model(
            inputs=disaster_predictor.inputs,
            outputs=disaster_predictor.get_layer("global_average_pooling2d").output)
# encoder.summary()

encodings, y_test = None, None
for tensor, y in testDS:
    encoding = encoder(tensor, training=False)
    if encodings is None:
        encodings = encoding
        y_test = y
    else:
        encodings = tf.concat([encodings, encoding], axis=0)
        y_test = tf.concat([y_test, y], axis=0)
    
print(encodings.shape)


"""T-SNE on Whole Test Dataset"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_embedded = TSNE(n_components=2,random_state=10).fit_transform(encodings)


new_unique_label = [i for i in range(len(unique_label))]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']


plt.figure(figsize=(7,6))
for i,label in enumerate(new_unique_label):
    inds = np.where(y_test==label, True, False)
    plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.05, color=colors[i]) #alpha=0.5

"""T-SNE on Hard Test Dataset"""
# restart tf graph
tf.keras.backend.clear_session()
backbone = get_backbone(size=416, class_num=7, backbone_h5='backbone.h5')
disaster_predictor = create_model()
augmentor = get_augmentor()
data = list(zip(test_img_paths, test_labels))
testDS = tf.data.Dataset.from_tensor_slices(data)
            
utils = TrainingUtils(data_root, unique_label)      
testDS = (testDS
	.map(utils.load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .map(utils.feature_extraction, num_parallel_calls=tf.data.AUTOTUNE)
	.batch(32)
	.prefetch(tf.data.AUTOTUNE)
)

disaster_predictor.load_weights(path)
encoder = tf.keras.models.Model(
            inputs=disaster_predictor.inputs,
            outputs=disaster_predictor.get_layer("global_average_pooling2d").output)

# predict
softmax_tensors = None
for tensor, _ in testDS:
    softmax_tensor = disaster_predictor(tensor, training=False)
    if softmax_tensors is None:
        softmax_tensors = softmax_tensor
    else:
        softmax_tensors = tf.concat([softmax_tensors, softmax_tensor], axis=0)
softmax_tensors = np.array(softmax_tensors)

# sort each of the softmax tensor
softmax_tensors.sort(axis=1)

# arg margin = P(x) of most probable - 2nd most probable
margin = softmax_tensors[:, -1]-softmax_tensors[:, -2]

# get the index to sort the softmax tensors
indices = np.argsort(margin)

# select the top N (number of query) of softmax tensors with largest margin
indices = list(indices[:1000])
mask = [True if i in indices else False for i in range(len(X_embedded))]
y_test = np.array(y_test)
print(y_test.shape)

# t-sne
X_embedded = X_embedded[mask]
y_test = y_test[indices]
for i,label in enumerate(new_unique_label):
    inds = np.where(y_test==label, True, False)
    plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.60, color=colors[i])
            
leg = plt.legend(unique_label)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.savefig('/home/tham/Desktop/demo.png', transparent=True, dpi=800)
plt.show()
