#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import os
import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an image file.')
    args.add_argument('-l', '--extension', type=str, default=None,
                      help='Optional. Required by the CPU Plugin for executing the custom operation on a CPU. '
                      'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--config', type=str, default=None,
                      help='Optional. Required by GPU or VPU Plugins for the custom operation kernel. '
                      'Absolute path to operation description file (.xml).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    # fmt: on
    return parser.parse_args()


def draw_outputs(img, outputs, class_names, label_only=True):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        if label_only:
            img = cv2.putText(img, '{}'.format(
                class_names[int(classes[i])]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        else:
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def main():  # noqa
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    if args.extension and args.device == 'CPU':
        log.info(f'Loading the {args.device} extension: {args.extension}')
        ie.add_extension(args.extension, args.device)

    if args.config and args.device in ('GPU', 'MYRIAD', 'HDDL'):
        log.info(f'Loading the {args.device} configuration: {args.config}')
        ie.set_config({'CONFIG_FILE': args.config}, args.device)

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('The sample supports only single input topologies')
        return -1

    if len(net.outputs) != 5:
        log.error('The sample supports models with 5 outputs')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get name of input blob
    input_blob = next(iter(net.input_info))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'
    
    precisions_of_outputs = ['FP16', 'FP16', 'FP16', 'FP16', 'FP16'] #['FP32', 'FP32', 'FP32', 'U16', 'U16']
    output_blobs = ['StatefulPartitionedCall/yolov3/disaster_head/reshape_1/Reshape', \
                    'StatefulPartitionedCall/yolov3/yolo_nms/ExpandDims', \
                    'StatefulPartitionedCall/yolov3/yolo_nms/ExpandDims_1', \
                    'StatefulPartitionedCall/yolov3/yolo_nms/ExpandDims_2', \
                    'StatefulPartitionedCall/yolov3/yolo_nms/ExpandDims_3']
    for precision, output_blob in zip(precisions_of_outputs, output_blobs):
        # print(output_blob+'\t'+precision)
        net.outputs[output_blob].precision = precision

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    original_image = cv2.imread(args.input)
    image = original_image.copy()
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape

    if image.shape[:-1] != (net_h, net_w):
        log.warning(f'Image {args.input} is resized from {image.shape[:-1]} to {(net_h, net_w)}')
        image = cv2.resize(image, (net_w, net_h))

    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)

    # ---------------------------Step 7. Do inference---------------------------------------------------------------------- UNTIL HERE
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: image})
    #for i in range(5):
        #print('\n')
        #print(res[output_blobs[i]])

    # ---------------------------Step 8. Custom Process output--------------------------------------------------------------------
    # Generate a label list
    labels = ['person']
    disaster_labels = ['earthquake', 'fire', 'flood', 'hurricane', 'landslide', \
                        'not_disaster', 'other_disaster']

    # get output image shape
    output_image = original_image.copy()
    h, w, _ = output_image.shape

    # get outputs
    disaster = res[output_blobs[0]]
    boxes = res[output_blobs[1]]
    scores = res[output_blobs[2]]
    classes = res[output_blobs[3]]
    nums = res[output_blobs[4]] # not sure why value is wrong (fix below)

    # fix nums
    nums = np.count_nonzero(scores[0] > 0)
    nums = np.array([nums])

    outputs = (boxes, scores, classes, nums)
    disaster = disaster_labels[np.argmax(disaster[0])]

    # draw outputs
    img = draw_outputs(original_image, (boxes, scores, classes, nums), labels)

    #img = cv2.putText(img, "Time: {:.2f}FPS".format(FPS), (0, 30),
                      #cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    img = cv2.putText(img, "Disaster: {}".format(disaster), (0, 50),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    img = cv2.putText(img, "Total victims: {}".format(nums[0]), (0, 70),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow('output', img)
    if cv2.waitKey(0) == ord('q'):
        pass    

    # ----------------------------------------------------------------------------------------------------------------------
    return 0


if __name__ == '__main__':
    sys.exit(main())