#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
from tqdm import tqdm
import glob
import re
import math

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

import numpy as np
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import (yolo_eval, yolo_body, tiny_yolo_body,
    tiny_yolo_infusion_body, infusion_layer, yolo_infusion_body, tiny_yolo_infusion_hydra_body,
    yolo_body_for_small_objs, tiny_yolo_small_objs_body)
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
# ap.add_argument("-w", "--weights",
#                 required=False,
#                 default=None,
#                 type=str,
#                 help="The weights to load the model. If not provided the trained_weights_final.h5 will be used from the logs dir.")
ap.add_argument("-e", "--only_epochs_above",
                required=False,
                default=None,
                type=int,
                help="Evaluate only epochs with its number above the specified integer. Otherwise, evaluate all weights")
ap.add_argument("-a", "--generate_all",
                required=False,
                action='store_true',
                help="Request the script to generate all output formats.")
ap.add_argument("-c", "--continue_version",
                required=False,
                default=None,
                type=str,
                help="The evaluation will skip inferences that are already done. The new inferences will use the given version.")
ap.add_argument("-ca", "--canonical_bboxes", required=False, action="store_true", help="The training configuration.")
ARGS = ap.parse_args()

train_config = None
with open(ARGS.config_path, 'r') as stream:
    train_config = yaml.load(stream)
print(train_config)

# if not train_config['log_dir'] in ARGS.weights:
#     raise Exception('Wrong setup: log_dir <-> weights')



class YOLO(object):
    def __init__(self, model_path=None):
        self.model_name = train_config['model_name']
        # self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        # self.model_path = 'logs/000_5epochs/trained_weights_final.h5'
        self.model_path = model_path
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
        if self.model_path:
            model_path = os.path.expanduser(self.model_path)
            assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        if self.model_name == 'tiny_yolo_infusion':
            print('Loading model weights', self.model_path)
            #old style
            # self.yolo_model = tiny_yolo_infusion_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            ## self.yolo_model.load_weights(self.model_path, by_name=True)
            #new style
            yolo_model, connection_layer = tiny_yolo_infusion_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            seg_output = infusion_layer(connection_layer)
            self.yolo_model = Model(inputs=yolo_model.input, outputs=[*yolo_model.output, seg_output])
            # self.yolo_model.load_weights(self.model_path, by_name=True)
        elif self.model_name == 'tiny_yolo_infusion_hydra':
            #old style
            self.yolo_model = tiny_yolo_infusion_hydra_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            # self.yolo_model.load_weights(self.model_path, by_name=True)
            #new style
            #not implemented yet
        elif self.model_name == 'yolo_infusion':
            print('Loading model weights', self.model_path)
            yolo_model, seg_output = yolo_infusion_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model = Model(inputs=yolo_model.input, outputs=[*yolo_model.output, seg_output])
            # self.yolo_model.load_weights(self.model_path, by_name=True)
        else:
            if self.model_name == 'yolo_small_objs':
                self.yolo_model = yolo_body_for_small_objs(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            elif self.model_name == 'tiny_yolo_small_objs':
                self.yolo_model = tiny_yolo_small_objs_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            else:
                try:
                    self.yolo_model = load_model(model_path, compile=False)
                except:
                    self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                        if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
                    # self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
                else:
                    assert self.yolo_model.layers[-1].output_shape[-1] == \
                        num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                        'Mismatch between model and given anchor and class sizes'
        if self.model_path:
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

def get_ratio(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (y_max - y_min)/(x_max - x_min)

def normal_round(number):
    #3.5
    integer_n = int(number) #3
    float_n = number - integer_n #0.5
    if float_n >= 0.5:
        return integer_n + 1 #4
    else:
        return integer_n

def get_canonical_bboxes(original_bboxes, img_width, img_height, round_type='normal', side_ajustment='one'):
    acceptable_ratios = [1,2,3]
    canonical_bboxes = []
    for bbox in original_bboxes:
        x_min, y_min, x_max, y_max, class_id = bbox
        new_x_min, new_y_min, new_x_max, new_y_max, _ = bbox

        if side_ajustment=='one':

            #step1: resize
            original_ratio = get_ratio(bbox)
            '''
            if the original ratio is higher than maximum acceptable_ratios, we need to increase the width.
            else we can increase the height.
            '''
            if original_ratio > max(acceptable_ratios):
                #we need to increase the width so that the height reduces to maximum acceptable_ratios.
                max_height_ratio = max(acceptable_ratios)
                original_height = y_max - y_min
                #the width should be increased to 1/max_height_ratio of the height.
                new_width = original_height // max_height_ratio
                #We need to expand it evenly in the sides.
                original_width = x_max - x_min
                width_diff = new_width - original_width #new_width is bigger.
                new_x_min = x_min - width_diff//2
                #We need to check if new_x_min still is in the image boundaries.
                if new_x_min <= 0:
                    #not enough space
                    new_x_min = 0
                    #we will expand the remaining to the other direction.
                new_x_max = new_x_min + new_width
                #lets check the same for the new_x_max
                if new_x_max >= img_width:
                    #not enough space
                    new_x_max = img_width
                    new_x_min = img_width - new_width
            else:
                #we can increase the height
                if round_type == 'normal':
                    new_height_ratio = normal_round(original_ratio) #normal rounding (up or down).
                elif round_type == 'up':
                    new_height_ratio = math.ceil(original_ratio) #rounding up
                new_height = math.ceil(new_height_ratio*(x_max - x_min)) # the ratio is relative to the width.
                #In how many pixels did the height grow?
                original_height = y_max - y_min
                height_diff = new_height - original_height
                #We need to split the growth up and down.
                #So, we put the ymin half the height_diff up.
                half_diff = height_diff // 2
                new_y_min = y_min - half_diff
                #But we check how many pixels are left upwards. We cannot overflow the img borders.
                if not (y_min - half_diff >= 0):
                    #not enough space.
                    new_y_min = 0
                #Now we have found the good new position for y_min, we add the complete needed height.
                new_y_max = new_y_min + new_height
                #We also need to check if we kept outselves the bottom image boundaries.
                if new_y_max >= img_height: #img_height does not include zero.
                    #We got out of space in the bottom. So lets move up the bbox to keep in the limits.
    #                 remaining_height = img_height - new_y_max
    #                 new_y_min -= remaining_height
    #                 new_y_max -= remaining_height
                    new_y_max = img_height
                    new_y_min = img_height - new_height

            #Lets check if we did it right, otherwise fallback to the original bbox.
            if not (new_x_min >= 0 and new_y_min >= 0 and new_x_max < img_width and new_y_max < img_height):
                # messed up, fallback!
    #             print('Could not convert the original bbox. We are going to use the original. Original: {}. Problematic: {}'.format(bbox, [new_x_min, new_y_min, new_x_max, new_y_max]))
                canonical_bboxes.append(bbox)
            else:
                canonical_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max, class_id])
        elif side_ajustment=='both':
            pass

    return canonical_bboxes

def detect_img(yolo,output_path):
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

    if ARGS.canonical_bboxes:
        result_detections = get_canonical_bboxes(result_detections, img_width=image.width, img_height=image.height)

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


    # yolo.close_session()

if __name__ == '__main__':

    weights_paths = glob.glob(os.path.join(train_config['log_dir'],'*.h5'))
    weights_paths.sort()
    output_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    yolo = YOLO()

    #get epoch number by regex
    #we could just split the string but lets avoid some future trouble.
    epoch_regex = re.compile('ep[0-9]{3}')

    for weight in weights_paths:
        #evaluate only epochs with number higher than the specified.
        if ARGS.only_epochs_above:
            m = epoch_regex.search(weight)
            if m:
                epoch_number = int(m.group().replace('ep',''))
                if epoch_number < ARGS.only_epochs_above:
                    print('Skipping weight:', weight)
                    #skip the current epoch weight.
                    continue
            else:
                '''
                if we dont recognize the epoch numbering pattern, we will
                evaluate the given weight just to be sure.
                The final_weight will match this.
                '''
                pass


        #infer_logdir_epochs_dataset_outputversion
        output_path = 'infer_{}_{}_{}_{}_{}_{}'.format(
            train_config['log_dir'].replace('/',''),
            os.path.basename(weight).split('-')[0], #[ep022]-loss5.235-val_loss5.453.h5
            train_config['dataset_name'],
            train_config['model_name'],
            train_config['short_comment'] if train_config['short_comment'] else '',
            ARGS.continue_version if ARGS.continue_version else output_version,
            )

        if ((train_config['dataset_name'] == 'pti01' and os.path.exists(output_path + '.txt')) or
            (train_config['dataset_name'] == 'caltech' and os.path.exists(output_path))):
                print('Skipping weights:', weight)
                continue
        else:
            print('Loading weights:', weight)
            yolo.yolo_model.load_weights(weight, by_name=True)
            detect_img(yolo,output_path=output_path)

    yolo.close_session()
