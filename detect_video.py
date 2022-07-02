import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2


# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
# flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
#                     'path to weights file')

flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('num_disaster_classes', 7, 'number of disaster classes in the model')
flags.DEFINE_float('iou_thresh', 0.5, 'iou threshold in range [0.5, 0.95]')
flags.DEFINE_float('score_thresh', 0.4, 'objectness threshold in range [0.1, 0.45]')

flags.DEFINE_string('video', 'demo_video.mp4', #demo_video
                    'path to video file or number for webcam)')

flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

flags.DEFINE_boolean('gpu', True, 'Use GPU or not')


def main(_argv):
    print(FLAGS.gpu)
    if not FLAGS.gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from yolov3_tf2.models import (
        YoloV3
    )
    from yolov3_tf2.dataset import transform_images
    from yolov3_tf2.utils import draw_outputs

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    mtl_model = YoloV3(FLAGS.size,
                        channels=3,
                        classes=FLAGS.num_classes, 
                        disaster_classes = FLAGS.num_disaster_classes,
                        yolo_iou_threshold=FLAGS.iou_thresh,
                        yolo_score_threshold=FLAGS.score_thresh,
                        auxiliary=True,
                        training=False
                      )

    mtl_model.load_weights('mtl_model.h5')
    # mtl_model.load_weights('mtl_model') # also can
    logging.info('weights loaded')

    # class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    class_names = ['person']
    disaster_class_names = ['earthquake', 'fire', 'flood', 'hurricane', 'landslide', \
                            'not_disaster', 'other_disaster']
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
    counter=0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            counter+=1
            if counter > 10: break
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums, disaster = mtl_model.predict(img_in)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        
        FPS = 1 / (sum(times)/len(times)*1000) * 1000
        disaster = disaster_class_names[tf.argmax(disaster[0]).numpy()]
        
        img = cv2.putText(img, "Time: {:.2f}FPS".format(FPS), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "Disaster: {}".format(disaster), (0, 50),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "Total victims: {}".format(nums[0]), (0, 70),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
