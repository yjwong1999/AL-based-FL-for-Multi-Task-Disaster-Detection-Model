import os
import timeit
import json

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from models import (
    YoloV3, YoloLoss, get_anchors,
    yolo_boxes, yolo_nms,
    Darknet,
    get_disaster_head
)
from utils import freeze_all

from models import YoloConv, YoloOutput, get_anchors
from tensorflow.keras.layers import Lambda

# define the flags
class Flags:
    def __init__(self):
        self.size = 416
        self.num_classes = 1
        self.yolo_max_boxes = 100
        self.yolo_iou_threshold = 0.5
        self.yolo_score_threshold = 0.5
        
        self.transfer = 'darknet'
        self.root = os.getcwd()
        self.ori_weight_path = os.path.join(self.root, 'data/yolov3.weights')
        self.new_weigth_path = os.path.join(self.root, 'checkpoints/yolov3.tf')
        
        self.epochs = 100
        self.mini_batch_size = 8
        self.num_grad_accumulates = 8 #8 #16
        self.learning_rate = 1e-3
        self.num_classes = 1

        self.mode = 'eager_fit'

FLAGS = Flags()

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
    
    return tf.keras.models.Model(inputs, outputs, name='yolov3')

backbone = get_backbone(size=416, class_num=7, backbone_h5='backbone.h5')
#victim_head = create_model()
#victim_head.load_weights("best_model.h5")
victim_head = tf.keras.models.load_model('best_model.h5')

# the model with box
model_with_box = YoloV3(FLAGS.size, 
               channels=3, 
               classes=FLAGS.num_classes, 
               auxiliary=False, 
               training=False,
               yolo_iou_threshold=0.5,
               yolo_score_threshold=0.4)
anchors, anchor_masks = get_anchors(FLAGS.size)

# load the pretrained model for transfer learning
YOLOV3_LAYER_LIST = [
        'yolo_conv_0',
        'yolo_conv_1',
        'yolo_conv_2',
        'yolo_output_0',
        'yolo_output_1',
        'yolo_output_2',
    ]

if FLAGS.transfer == 'darknet':   
    # transfer the yolo darknet backbone weights to our current model
    model_with_box.get_layer('yolo_darknet').set_weights(
        backbone.get_layer('yolo_darknet').get_weights())
    for layer in YOLOV3_LAYER_LIST:
        model_with_box.get_layer(layer).set_weights(
            victim_head.get_layer(layer).get_weights())
        
        # freeze the yolo darknet backbone 
        freeze_all(model_with_box.get_layer(layer))

# get anchor
model_with_box.save("model_with_box.h5")
anchors, anchor_masks = get_anchors(FLAGS.size)

# data root
data_root = '/home/tham/Documents/fyp_yijie/crisis_vision_benchmarks/'
data_root = tf.convert_to_tensor(data_root, tf.string)

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
            annot += [[0, 0, 0, 0, 0]] * (FLAGS.yolo_max_boxes - len(annot))
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

print(len(train_annots))
print(len(val_annots))
print(len(test_annots))

def predict_annotation(img_paths, pred_json_path, name):
    if not os.path.isfile(pred_json_path):
        # start predicting + annotation
        print('starting...')
        t1 = timeit.default_timer()
        
        # a dict to store all paths and all annotation
        annot_json  = {
            'name': None,
            'data': None,
        }

        # a list to store all annotation
        all_annot = []

        # iterate all images
        for img_path in img_paths:
            # load image
            new_img_path = tf.strings.join([data_root, img_path])
            ori_img = tf.io.read_file(new_img_path)
            ori_img = tf.image.decode_jpeg(ori_img, channels=3)
            img = tf.image.resize(ori_img, (FLAGS.size, FLAGS.size)) / 255.0

            # get the prediction
            boxes, scores, classes, nums = model_with_box.predict(np.expand_dims(img, 0))
            boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

            # the template for box cornet: x1, y1, x2, y2
            box_corner = {
                'x1':0,
                'y1':0,
                'x2':0,
                'y2':0,
            }

            # a list to store all bbox for an image
            all_box_in_an_image = []

            # a unit of annot
            annot_unit = {
                'img_path': None,
                'img_shape': None,
                'bboxs': None,
                'confidences': None,
            }

            # start recording
            for i in range(nums):
                # record the box corner coord values
                box = boxes[i]
                x1, y1, x2, y2 = box
                box_corner['x1'] = str(x1)
                box_corner['y1'] = str(y1)
                box_corner['x2'] = str(x2)
                box_corner['y2'] = str(y2)
                # record this box corner
                all_box_in_an_image.append(box_corner.copy())

            # record all bbox for this image
            annot_unit['img_path'] = img_path.numpy().decode('utf-8')
            annot_unit['img_shape'] = [str(i) for i in list(ori_img.shape)]
            annot_unit['bboxs'] = all_box_in_an_image.copy()
            annot_unit['confidences'] = [str(i) for i in list(scores[:nums])]
            if nums == 0:
                annot_unit['bboxs'] = None
                annot_unit['confidences'] = None

            all_annot.append(annot_unit)

        # save all information in the annoation dictionary
        annot_json['name'] = name
        annot_json['data'] = all_annot

        with open(pred_json_path, "w") as out_file:
            json.dump(annot_json, out_file, indent = 6) 
        
        t2 = timeit.default_timer()
        print('Total time to annotate "{}": {} min\n'.format(pred_json_path, (t2-t1)/60))
    
    else:
        print('JSON file "{}" already existed\n'.format(pred_json_path))
        
pred_train_json_path = 'others/yolo_pred_train_damage_severity_person.json'
pred_val_json_path = 'others/yolo_pred_val_damage_severity_person.json'
pred_test_json_path = 'others/yolo_pred_test_damage_severity_person.json'

test_img_paths = tf.convert_to_tensor(test_img_paths, dtype=tf.string)
predict_annotation(test_img_paths, pred_test_json_path, 'testing')

iou_thresh = 0.5

def get_iou(img_shape, bbox1, bbox2):
    '''
    bbox 1 should be left box
    bbox 2 should be right box
    '''
    # make sure the box are correct 
    if bbox1['x1'] > bbox2['x1']:
        bbox1, bbox2 = bbox2, bbox1
        #print('terbalik')
        
    # unpack the variables
    row, col, _ = img_shape 
    x1 = bbox1['x1']
    y1 = bbox1['y1']
    x2 = bbox1['x2']
    y2 = bbox1['y2']

    xA = bbox2['x1']
    yA = bbox2['y1']
    xB = bbox2['x2']
    yB = bbox2['y2']
  
    # scale the coord to original size
    x1, x2, xA, xB = x1 * col, x2 * col, xA * col, xB * col
    y1, y2, yA, yB = y1 * row, y2 * row, yA * row, yB * row
    
    # round down the decimal place
    x1, x2, xA, xB = int(x1), int(x2), int(xA), int(xB)
    y1, y2, yA, yB = int(y1), int(y2), int(yA), int(yB)
    
    # the condition where the two bbox intersect
    no_overlap = (x2 < xA or xB < x1) or (y2 < yA or yB < y1)
    if not no_overlap:
        X = [x1, x2, xA, xB]
        Y = [y1, y2, yA, yB]
        X.sort()
        Y.sort()
        area_intersect = (X[2] - X[1]) * (Y[2] - Y[1])
        area_bbox1 = (x2-x1) * (y2-y1)
        area_bbox2 = (xB-xA) * (yB-yA)
        iou = area_intersect / (area_bbox1 + area_bbox2 - area_intersect)
        return iou
    
    # else there is not intersection
    return 0

def count_TP_FP(img_shape, actual_boxes, pred_boxes):
    # initialize variables
    positives = [False] * len(pred_boxes)
    
    # loop all actual boxes
    for actual_box in actual_boxes:
        # loop all pred_boxes
        idxs = []
        ious = []
        for i in range(len(pred_boxes)):
            if positives[i] == False:
                #print(actual_box)
                #print(pred_boxes[i])
                iou = get_iou(img_shape, actual_box, pred_boxes[i])
                if iou > iou_thresh: 
                    idxs.append(i)
                    ious.append(iou)
        # determine if the boxes are TP or FP
        if len(idxs) == 1:
            positives[idxs[0]] = True
        elif len(idxs) > 1:
            max_iou = max(ious)
            max_iou_idx = ious.index(max_iou)
            positives[max_iou_idx] = True
    # return positives
    return positives    
 
def process_an_image(actual_annot, pred_annot):
    # get the variables
    actual_boxes = actual_annot['bboxs']
    for i in range(len(actual_boxes)):
        actual_boxes[i]['x1'] = float(actual_boxes[i]['x1'])
        actual_boxes[i]['y1'] = float(actual_boxes[i]['y1'])
        actual_boxes[i]['x2'] = float(actual_boxes[i]['x2'])
        actual_boxes[i]['y2'] = float(actual_boxes[i]['y2'])

    pred_boxes = pred_annot['bboxs']
    if pred_boxes is not None:
        for i in range(len(pred_boxes)):
            pred_boxes[i]['x1'] = float(pred_boxes[i]['x1'])
            pred_boxes[i]['y1'] = float(pred_boxes[i]['y1'])
            pred_boxes[i]['x2'] = float(pred_boxes[i]['x2'])
            pred_boxes[i]['y2'] = float(pred_boxes[i]['y2'])
        pred_confs = [float(i) for i in pred_annot['confidences']]
    else:
        pred_boxes = []
        pred_confs = []
        
    img_shape = [float(i) for i in pred_annot['img_shape']]
    
    positives = count_TP_FP(img_shape, actual_boxes, pred_boxes)
    return positives, pred_confs, len(actual_boxes)

def get_precision_recall(all_positives, ground_truth_counts):
    precisions = []
    recalls = []
    TP = 0
    FP = 0
    for positive in all_positives:
        if positive:
            TP += 1
        else:
            FP += 1
        precisions.append(TP/(TP+FP))
        recalls.append(TP/ground_truth_counts)
    
    return np.array(precisions), np.array(recalls)

def get_curve(actual_annot_json, pred_annot_json, MAX_COUNT=None):
    '''
    make sure the sequence correct
    '''
    # extract the actual annotation (bbox corner)
    with open(actual_annot_json) as f:
        # load the dataset
        json_dataset = json.load(f)
        # get data
        actual_annots = json_dataset['data']
    
    # extract the predicted annotation (bbox corner and confidence)
    with open(pred_annot_json) as f:
        # load the dataset
        json_dataset = json.load(f)
        # get data
        pred_annots = json_dataset['data']    
    
    # loop all data
    count=0
    if MAX_COUNT is None:
        MAX_COUNT = len(pred_annots)

    all_positives = []
    all_pred_confs = []
    ground_truth_counts = 0
    for actual_annot, pred_annot in zip(actual_annots, pred_annots):
        positives, pred_confs,  ground_truth_count = process_an_image(actual_annot, pred_annot)
        # all_positives.append(positives)
        # all_pred_confs.append(pred_confs)
        all_positives += positives
        all_pred_confs += pred_confs
        ground_truth_counts += ground_truth_count
        count += 1
        if count == MAX_COUNT: break
    
    # sort all_positives & all_pred_confs in descending order of confidence
    sort_idx = np.argsort(all_pred_confs)
    sort_idx = np.flip(sort_idx)
    all_positives = np.array(all_positives)[sort_idx]
    all_pred_confs = np.array(all_pred_confs)[sort_idx]
    
    # get precisions and recalls
    precisions, recalls = get_precision_recall(all_positives, ground_truth_counts)

    return precisions, recalls

# def plot_curve(color='blue'):
#     plt.plot(recalls, precisions, linewidth=4, color=color, zorder=0, alpha=0.6)

#     plt.xlabel("Recall", fontsize=12, fontweight='bold')
#     plt.ylabel("Precision", fontsize=12, fontweight='bold')
#     plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
#     plt.xlim([0, 1.0])
#     plt.ylim([0.88, 1.0])
#     plt.fill_between(recalls, precisions, y2=0, alpha=0.2)
#     plt.text(0.8, 0.98, 'mAP:\n{:.2f}%'.format(AP*100), fontsize=14)
#     plt.show()

plt.figure(figsize=(12,8))
precisions, recalls = get_curve(test_json_path, pred_test_json_path)
AP = np.sum((recalls[1:] - recalls[:-1]) * precisions[:-1])
print(AP)
plt.plot(recalls, precisions, linewidth=4, color='green', zorder=0, alpha=0.6, label='Testing PR Curve')
#plt.fill_between(recalls, precisions, y2=0, alpha=0.2)

plt.xlabel("Recall", fontsize=15, fontweight='bold')
plt.ylabel("Precision", fontsize=15, fontweight='bold')
plt.title("Precision-Recall (PR) Curve", fontsize=15, fontweight="bold")
# plt.xlim([0, 1.0])
plt.legend(fontsize=12)
plt.show()

