# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:54:37 2022

@author: e-default
"""
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import json
from typing import List

import tensorflow as tf

from models import YoloV3, YoloConv, YoloOutput, yolo_boxes, yolo_nms, get_anchors
from tensorflow.keras.layers import Lambda
from utils import freeze_all, draw_outputs

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)

"""Image Paths and BBox"""
print('Getting Data')
def get_annotation(json_path, max_limit):
    with open(json_path) as f:
        # load the dataset
        json_dataset = json.load(f)
        # get data
        img_paths = []
        annots = []
        count = 0
        if max_limit is None:
            max_limit = len(json_dataset['data'])
        for data in json_dataset['data']:
            # image path
            img_paths.append(data['img_path'])
            # get the box corner (x1, y1, x2, y2)
            bboxs = data['bboxs']
            annot = []
            for bbox in bboxs:
                x1 = bbox['x1']
                y1 = bbox['y1']
                x2 = bbox['x2']
                y2 = bbox['y2']
                # last 0 is class for person
                annot += [[float(x1), float(y1), float(x2), float(y2), 0.0]]
            annot += [[0, 0, 0, 0, 0]] * (100 - len(annot))
            annot = tf.convert_to_tensor(annot)
            annots.append(annot)
            count += 1
            if count == max_limit:
                break
                
    return img_paths, annots

MAX_LIMIT = None
train_json_path = 'others/train_damage_severity_person.json'
val_json_path = 'others/val_damage_severity_person.json'
test_json_path = 'others/test_damage_severity_person.json'

train_img_paths, train_annots = get_annotation(train_json_path, MAX_LIMIT)
val_img_paths, val_annots = get_annotation(val_json_path, MAX_LIMIT)
test_img_paths, test_annots = get_annotation(test_json_path, MAX_LIMIT)

train_img_paths = tf.convert_to_tensor(train_img_paths, dtype=tf.string)
train_annots = tf.convert_to_tensor(train_annots, dtype=tf.float32)
val_img_paths = tf.convert_to_tensor(val_img_paths, dtype=tf.string)
val_annots = tf.convert_to_tensor(val_annots, dtype=tf.float32)
test_img_paths = tf.convert_to_tensor(test_img_paths, dtype=tf.string)
test_annots = tf.convert_to_tensor(test_annots, dtype=tf.float32)

print(len(train_annots))
print(len(val_annots))
print(len(test_annots))

# convert data root to tf string
data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
data_root = tf.convert_to_tensor(data_root, tf.string)

"""model defintition"""
# get the backbone
def get_backbone(size, class_num, backbone_h5):
    # define input
    in_shape = (size, size, 3)
    input_images = tf.keras.layers.Input(shape=in_shape)
    x = input_images
    # load the backbone
    backbone = tf.keras.models.load_model(backbone_h5)
    backbone.trainable = False
    x_31, x_61, x_91 = backbone(x)
    # define the model
    model = tf.keras.models.Model(input_images, (x_31, x_61, x_91), name='backbone')
    return model
  
backbone = get_backbone(size=416, class_num=7, backbone_h5='backbone.h5')


"""Training Utilities"""
class TrainingUtils:
    def __init__(self, data_root):
        self.data_root = data_root
      
    @tf.function
    def _augment(self, image):
        # brightness
        image = tf.image.random_brightness(image, 0.2)
        # contrast
        image = tf.image.random_contrast(image, 0.8, 1.50)
        # saturation
        image = tf.image.random_saturation(image, 0.80, 1.20) #ori is 0.75-1.25
        # return the image and the label
        return image
    
    def load_image(self, image_path):
        # read the image from disk, decode it, resize it, and scale the
        # pixels intensities to the range [0, 1]
        image_path = tf.strings.join([self.data_root, image_path])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416)) / 255.0
        # return the image and the integer encoded label
        return image
      
    def load_image_augment(self, image_path):
        # read the image from disk, decode it, resize it, and scale the
        # pixels intensities to the range [0, 1]
        image_path = tf.strings.join([self.data_root, image_path])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416)) / 255.0
        # augment
        image = self._augment(image)
        # return the image and the integer encoded label
        return image
    
    @tf.function
    def _transform_targets_for_output(self, y_true, grid_size, anchor_idxs):
        # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
        N = tf.shape(y_true)[0]
    
        # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
        y_true_out = tf.zeros(
            (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    
        anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    
        indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        idx = 0
        for i in tf.range(N):
            for j in tf.range(tf.shape(y_true)[1]):
                if tf.equal(y_true[i][j][2], 0):
                    continue
                anchor_eq = tf.equal(
                    anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
    
                if tf.reduce_any(anchor_eq):
                    box = y_true[i][j][0:4]
                    box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
    
                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                    grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
    
                    # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                    indexes = indexes.write(
                        idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                    updates = updates.write(
                        idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                    idx += 1
    
        return tf.tensor_scatter_nd_update(
            y_true_out, indexes.stack(), updates.stack())

    def transform_targets(self, y_train, anchors, anchor_masks, size):
        y_train = tf.expand_dims(y_train, axis=0) ####################################
        y_outs = []
        grid_size = size // 32
        # calculate anchor index for true boxes
        anchors = tf.cast(anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                         (1, 1, tf.shape(anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
            tf.minimum(box_wh[..., 1], anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    
        y_train = tf.concat([y_train, anchor_idx], axis=-1)
        for anchor_idxs in anchor_masks:
            y_out = self._transform_targets_for_output(
                y_train, grid_size, anchor_idxs)
            #y_out = tf.squeeze(y_out, axis=0) ####################################
            y_outs.append(y_out) 
            grid_size *= 2
    
        return tuple(y_outs)
      
utils = TrainingUtils(data_root)
anchors, anchor_masks = get_anchors(416)


"""Mnist Shard Descriptor."""
class MnistShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Initialise the dataset"""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]
        self.augment = data_type=="train" #if train then we augment

    def __getitem__(self, index: int):
        """Return an item by the index."""
        # get data
        image_paths, annots = None, None
        for idx in index:
            image_path = tf.expand_dims(self.x[idx], axis=0)
            annot = tf.expand_dims(self.y[idx], axis=0)
            if image_paths is None:
                image_paths = image_path
                annots = annot
            else:
                image_paths = tf.concat([image_paths, image_path], axis=0)
                annots = tf.concat([annots, annot], axis=0)
                
        # load image
        tensors_0 = None
        tensors_1 = None
        tensors_2 = None
        for i, image_path in enumerate(image_paths):
            # augment image
            if self.augment:
                image = utils.load_image_augment(image_path)
            else:
                image = utils.load_image(image_path)
            # feature extraction
            image = tf.expand_dims(image, axis=0)
            tensor = backbone(image, training=False)
            if tensors_0 is None: #if any of the tensors are empty
                tensors_0 = tensor[0]
                tensors_1 = tensor[1]
                tensors_2 = tensor[2]
            else:
                tensors_0 = tf.concat([tensors_0, tensor[0]], axis=0)
                tensors_1 = tf.concat([tensors_1, tensor[1]], axis=0)
                tensors_2 = tf.concat([tensors_2, tensor[2]], axis=0)
        tensors = (tensors_0, tensors_1, tensors_2)    

        # encode the label
        annots_0 = None
        annots_1 = None
        annots_2 = None
        for annot in annots:
            # annot = np.array(annot, np.float32)
            annot = utils.transform_targets(annot, anchors, anchor_masks, 416)            
            if annots_0 is None: #if any of the tensors are empty
                annots_0 = annot[0]
                annots_1 = annot[1]
                annots_2 = annot[2]
            else:
                annots_0 = tf.concat([annots_0, annot[0]], axis=0)
                annots_1 = tf.concat([annots_1, annot[1]], axis=0)
                annots_2 = tf.concat([annots_2, annot[2]], axis=0)
        annots = (annots_0, annots_1, annots_2)           

        # return the image and the integer encoded label
        #print(images.shape, labels.shape)
        #print(index)
        return tensors, annots

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)



class MnistShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.data_by_type = {
            'train': (train_img_paths, train_annots),
            'val': (test_img_paths, test_annots)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MnistShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['416', '416', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Mnist dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
