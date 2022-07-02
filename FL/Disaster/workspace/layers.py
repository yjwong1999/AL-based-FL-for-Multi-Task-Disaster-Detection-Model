# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Useful keras functions."""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
import numpy as np

from models import get_disaster_head

def create_model():
    x = np.ones((1, 26, 26, 512))
    x = tf.convert_to_tensor(x)
    disaster_head = get_disaster_head(x, class_num=7)
    return disaster_head


# Instantiate an optimizer.
# Instantiate a loss function.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=100.0, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
'''
train_acc_metric = [tf.keras.metrics.SparseCategoricalAccuracy(), \
                            tf.keras.metrics.Precision(), \
                            tf.keras.metrics.Recall(), \
                            tfa.metrics.F1Score(7)]
val_acc_metric = [tf.keras.metrics.SparseCategoricalAccuracy(), \
                            tf.keras.metrics.Precision(), \
                            tf.keras.metrics.Recall(), \
                            tfa.metrics.F1Score(7)]
'''
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
