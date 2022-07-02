# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Useful keras functions."""
import os
import tensorflow as tf
from models import YoloV3, YoloConv, YoloOutput, yolo_boxes, yolo_nms, get_anchors, YoloLoss
from tensorflow.keras.layers import Lambda
from utils import freeze_all, draw_outputs

# define victim head
def create_model(size=416, classes=1, training=True, score_threshold=0.5):
    # get anchors
    anchors, masks = get_anchors(size)
    
    # declare input
    x_36_shape = (None, None, 256)
    x_61_shape = (None, None, 512)
    x_91_shape = (None, None, 1024)
    
    x_36 = tf.keras.layers.Input(x_36_shape, name='input_0')
    x_61 = tf.keras.layers.Input(x_61_shape, name='input_1')
    x_91 = tf.keras.layers.Input(x_91_shape, name='input_2')
    inputs = (x_36, x_61, x_91)

    x = YoloConv(512, name='yolo_conv_0')(x_91)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return tf.keras.models.Model((x_36, x_61, x_91), (output_0, output_1, output_2), name='victim_head')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes, yolo_iou_threshold=0.5,
                                        yolo_score_threshold=score_threshold),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    
    return tf.keras.models.Model(inputs, outputs, name='victim_head')


"""Optimizer"""
optimizer = tf.keras.optimizers.Adam(
    learning_rate=100.0
)


"""Loss"""
anchors, anchor_masks = get_anchors(416)
loss = [YoloLoss(anchors[mask], classes=1)
            for mask in anchor_masks]
