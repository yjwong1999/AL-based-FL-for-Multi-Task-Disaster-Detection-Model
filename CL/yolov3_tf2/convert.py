import numpy as np
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf
        
def convert_weight(ori_weight_path, new_weigth_path, size):
    # define the num_classes of the model to be loaded
    num_classes = 80
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3(size=None, channels=3, classes=num_classes, auxiliary=False, training=False)

    load_darknet_weights(yolo, ori_weight_path)
    img = np.random.random((1, size, size, 3)).astype(np.float32)
    output = yolo(img)

    yolo.save_weights(new_weigth_path)