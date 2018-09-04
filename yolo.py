#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
from tqdm import tqdm


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

import numpy as np
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, tiny_yolo_infusion_body, infusion_layer
from yolo3.utils import letterbox_image
import os,datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=1

import argparse
import yaml
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--config_path",
                required=True,
                default=None,
                type=str,
                help="The training configuration.")
ap.add_argument("-w", "--weights",
                required=False,
                default=None,
                type=str,
                help="The weights to load the model. If not provided the trained_weights_final.h5 will be used from the logs dir.")
ap.add_argument("-a", "--generate_all",
                required=False,
                action='store_true',
                help="Request the script to generate all output formats.")
ARGS = ap.parse_args()

train_config = None
with open(ARGS.config_path, 'r') as stream:
    train_config = yaml.load(stream)
print(train_config)

output_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#infer_logdir_epochs_dataset_outputversion
output_path = 'infer_{}_{}_{}_{}_{}_{}'.format(
    train_config['log_dir'].replace('/',''),
    os.path.basename(ARGS.weights).split('-')[0], #[ep022]-loss5.235-val_loss5.453.h5
    train_config['dataset_name'],
    train_config['model_name'],
    train_config['short_comment'] if train_config['short_comment'] else '',
    output_version,
    )

class YOLO(object):
    def __init__(self):
        self.model_name = train_config['model_name']
        # self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        # self.model_path = 'logs/000_5epochs/trained_weights_final.h5'
        self.model_path = ARGS.weights
        print(self.model_path)

        # self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = train_config['classes_path']
        # self.classes_path = 'model_data/coco_classes.txt'
        self.anchors_path = train_config['anchors_path']
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        # self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.model_image_size = (480,640) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        if self.model_name == 'tiny_yolo_infusion':
            yolo_model, connection_layer = tiny_yolo_infusion_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            seg_output = infusion_layer(connection_layer)
            self.yolo_model = Model(inputs=yolo_model.input, outputs=[*yolo_model.output, seg_output])
            self.yolo_model.load_weights(self.model_path, by_name=True)
        else:
            try:
                self.yolo_model = load_model(model_path, compile=False)
            except:
                self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                    if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
                self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            else:
                assert self.yolo_model.layers[-1].output_shape[-1] == \
                    num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                    'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou, model_name=self.model_name)
        return boxes, scores, classes

    def detect_image(self, image, verbose=False, draw=False, output_file=None):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        if verbose:
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        if draw:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

        detections = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if draw:
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if verbose:
                print(label, (left, top), (right, bottom))

            if draw:
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            # <left> <top> <right> <bottom> <class_id> <confidence>
            detections.append([left, top, right, bottom, c, score])

        end = timer()
        if verbose:
            print('Executed in: ', end - start)

        return image, detections

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    result_detections = []
    result_images = []

    test_annotations = train_config['test_path']
    with open(test_annotations,'r') as annot_f:
            for annotation in tqdm(annot_f):
                try:
                    # print(annotation)
                    # image = Image.open('/home/grvaliati/workspace/datasets/pti/PTI01/C_BLC03-02/0/18/01/08/16/57/18/00150-capture.jpg')
                    img_path = annotation.split(' ')[0].strip()
                    # print('img_path',img_path)
                    image = Image.open(img_path)
                except Exception as e:
                    print('Error while opening file.', e)
                    break;
                else:
                    r_image, detections = yolo.detect_image(image)
                    result_images.append(r_image.filename)
                    result_detections.append(detections)
                    # r_image.show()
                    # r_image.save('img_seg_test.jpg')

    if ARGS.generate_all or train_config['dataset_name'] == 'pti01':
        print('Saving results for ',train_config['dataset_name'])

        pti01_output_path = output_path + '.txt'
        print('Saving in ', pti01_output_path)

        with open(pti01_output_path, 'w') as output_f:
            for index, image_filename in enumerate(result_images):
                detections_string = ''
                for d in result_detections[index]:
                    # <left> <top> <right> <bottom> <class_id> <confidence>
                    detections_string += ' {},{},{},{},{},{}'.format(d[0], d[1], d[2], d[3], d[4], d[5])

                output_f.write('{}{}\n'.format(image_filename, detections_string))

    if ARGS.generate_all or train_config['dataset_name'] == 'caltech':
        print('Saving results for ',train_config['dataset_name'])
        print('Saving in ', output_path)

        for index, image_filename in enumerate(result_images):
            #image_filename /absolute/path/set00_V000_662.jpg
            image_name = os.path.basename(image_filename) #set00_V000_662.jpg
            path_elements = image_name.replace('.jpg','').split('_')
            annot_dir = os.path.join(path_elements[0],path_elements[1])
            annot_dir = os.path.join(output_path,annot_dir)
            os.makedirs(annot_dir, exist_ok=True)
            #annot file format -> "I00029.txt"
            annot_name = 'I{}.txt'.format(path_elements[2].zfill(5))
            annot_filename = os.path.join(annot_dir, annot_name)
            with open(annot_filename, 'w') as output_f:
                for d in result_detections[index]:
                    #caltech evaluation format -> "[left, top, width, height, score]".
                    left, top, right, botton, class_id, score = d[0], d[1], d[2], d[3], d[4], d[5]
                    width = right - left
                    height = botton - top
                    output_f.write('{},{},{},{},{}\n'.format(left,top,width,height,score))


    yolo.close_session()

if __name__ == '__main__':
    detect_img(YOLO())
