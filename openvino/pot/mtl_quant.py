import json
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

import addict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torchmetrics
from compression.api import DataLoader, Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, load_model
from compression.pipeline.initializer import create_pipeline
from openvino.runtime import Core
from yaspin import yaspin
import tensorflow as tf

# sys.path.append("../utils")
# from notebook_utils import benchmark_model


"""Load the OpenVINO IR Model"""
ir_path = 'model/mtl/saved_model.xml'
bin_path = 'model/mtl/saved_model.bin'
ie = Core()
model = ie.read_model(model=ir_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))
input_size = input_layer.shape
_, _, input_height, input_width = input_size


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

# train_img_paths = tf.convert_to_tensor(train_img_paths, dtype=tf.string)
train_annots = tf.convert_to_tensor(train_annots, dtype=tf.float32)
# val_img_paths = tf.convert_to_tensor(val_img_paths, dtype=tf.string)
val_annots = tf.convert_to_tensor(val_annots, dtype=tf.float32)
# test_img_paths = tf.convert_to_tensor(test_img_paths, dtype=tf.string)
test_annots = tf.convert_to_tensor(test_annots, dtype=tf.float32)

print(len(train_annots))
print(len(val_annots))
print(len(test_annots))

# convert data root to tf string
data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
#data_root = tf.convert_to_tensor(data_root, tf.string)


"""Data Loader for OpenVINO"""
class DetectionDataLoader(DataLoader):
    def __init__(self, x, y, target_size):
        self.image_paths = x[:200]
        self.annots = y[:200]
        self.target_size = target_size
        

    def __getitem__(self, index):
        image_path = data_root + self.image_paths[index]
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, self.target_size)

        target_annotations = ["To be Decide"]
        item_annotation = (index, target_annotations)
        input_image = np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(
            np.float32
        )
        return (
            item_annotation,
            input_image,
        )

    def __len__(self):
        return len(self.image_paths)

      
"""Configuration"""
# Model config specifies the model name and paths to model .xml and .bin file
model_config = addict.Dict(
    {
        "model_name": "mtl",
        "model": ir_path,
        "weights": bin_path,
    }
)

# Engine config
engine_config = addict.Dict({"device": "CPU"})

# Standard DefaultQuantization config. For this tutorial stat_subset_size is ignored
# because there are fewer than 300 images. For production use 300 is recommended.
default_algorithms = [
    {
        "name": "DefaultQuantization",
        "stat_subset_size": 300,
        "params": {
            "target_device": "ANY",
            "preset": "mixed",  # choose between "mixed" and "performance"
        },
    }
]

print(f"model_config: {model_config}")
      

"""POT"""
# Step 1: create data loader
data_loader = DetectionDataLoader(
    test_img_paths, test_annots, target_size=(input_width, input_height)
)

# Step 2: load model
ir_model = load_model(model_config=model_config)

# Step 3: initialize the metric
# For DefaultQuantization, specifying a metric is optional: metric can be set to None
metric = None

# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

# Step 5: Create a pipeline of compression algorithms.
# algorithms is defined in the Config cell above this cell
pipeline = create_pipeline(default_algorithms, engine)

# Step 6: Execute the pipeline to quantize the model
algorithm_name = pipeline.algo_seq[0].name
with yaspin(
    text=f"Executing POT pipeline on {model_config['model']} with {algorithm_name}"
) as sp:
    start_time = time.perf_counter()
    compressed_model = pipeline.run(ir_model)
    end_time = time.perf_counter()
    sp.ok("✔")
print(f"Quantization finished in {end_time - start_time:.2f} seconds")

# Step 7 (Optional): Compress model weights to quantized precision
#                    in order to reduce the size of the final .bin file
compress_model_weights(compressed_model)

# Step 8: Save the compressed model to the desired path.
# Set save_path to the directory where the compressed model should be stored
preset = pipeline._algo_seq[0].config["preset"]
algorithm_name = pipeline.algo_seq[0].name
compressed_model_paths = save_model(
    model=compressed_model,
    save_path="optimized_model",
    model_name=f"{ir_model.name}_{preset}_{algorithm_name}",
)

compressed_model_path = compressed_model_paths[0]["model"]
print("The quantized model is stored at", compressed_model_path)