"""
Retrain the YOLO model for your own dataset.
"""

import argparse
import yaml
from raven import Client
import datetime
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--config_path",
                required=True,
                default=None,
                type=str,
                help="The training configuration.")
ap.add_argument("-m", "--memory",
                required=False,
                default=None,
                type=float,
                help="The amount of memory to be used by the framework in MB.")
ARGS = ap.parse_args()

TOTAL_GPU_MEMORY=12118
if ARGS.memory:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = ARGS.memory / TOTAL_GPU_MEMORY
    set_session(tf.Session(config=config))

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, tiny_yolo_infusion_body, infusion_layer, tiny_yolo_seg_body
from yolo3.utils import get_random_data





def _main(train_config):
    # annotation_path = 'train.txt'
    # annotation_path = 'train_pti01_6342imgs_v20180706193526_keras.txt'
    annotation_path = train_config['train_path']
    # log_dir = 'logs/000/'
    log_dir = train_config['log_dir']
    # classes_path = 'model_data/voc_classes.txt'
    # classes_path = 'model_data/pti_classes.txt'
    classes_path = train_config['classes_path']
    # anchors_path = 'model_data/yolo_anchors.txt'
    anchors_path = train_config['anchors_path']
    model_name = train_config['model_name']
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    freeze_body = 1
    pretrained_weights_path = train_config['pretrained_weights_path']

    input_shape = (416,416) # multiple of 32, hw
    # input_shape = (480,640)

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True,
            freeze_body=freeze_body, weights_path=pretrained_weights_path, model_name=model_name)
    else:
        raise Exception('Unknown model.')

    logging = TensorBoard(log_dir=log_dir, write_grads=True, write_images=True)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    seg_shape = (13,13)
    # seg_shape = (15,20)
    # seg_shape = (26,26)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    batch_size_freezed = 4
    epochs_freezed = 0
    if True and epochs_freezed > 0:
        compile_model(model, model_name)
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_freezed))
        model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size_freezed, input_shape, anchors, num_classes, model_name, seg_shape=seg_shape),
                steps_per_epoch=max(1, num_train//batch_size_freezed),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size_freezed, input_shape, anchors, num_classes, model_name, seg_shape=seg_shape),
                validation_steps=max(1, num_val//batch_size_freezed),
                epochs=epochs_freezed,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        #Unfreeze all layers.
        print('Unfreeze all of the layers.')
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        compile_model(model, model_name)

        batch_size_unfreezed = 4 # note that more GPU memory is required after unfreezing the body
        epochs_unfreezed = 50
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfreezed))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size_unfreezed, input_shape, anchors, num_classes, model_name, seg_shape=seg_shape),
            steps_per_epoch=max(1, num_train//batch_size_unfreezed),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size_unfreezed, input_shape, anchors, num_classes, model_name, seg_shape=seg_shape),
            validation_steps=max(1, num_val//batch_size_unfreezed),
            epochs=epochs_freezed + epochs_unfreezed,
            initial_epoch=epochs_freezed,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.

def compile_model(model, model_name):
    print('model_name',model_name)
    if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred, #I guess this is a dump operation. Does nothing
                'seg_output' : 'categorical_crossentropy'
            }) # recompile to apply the change
    elif model_name in ['tiny_yolo', 'yolo']:
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred,
            }) # recompile to apply the change
    elif model_name in ['tiny_yolo_seg']:
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss={
                'seg_output' : 'categorical_crossentropy'
            })
    else:
        raise Exception('The model_name is unknown: ', model_name)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', model_name=None):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    if model_name == 'yolo_infusion':
        raise Exception('not implemented yet.')
    else:
        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, num_classes+5)) for l in range(3)]

        model_body = yolo_body(image_input, num_anchors//3, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers)-3)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5', model_name=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    if model_name == 'tiny_yolo_seg':

        model_body = tiny_yolo_seg_body(image_input, num_anchors//2, num_classes)
        print('Create Tiny YOLOv3 INFUSION model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            # raise Exception('freezing requires review.')
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        print('*model_body.output', model_body.output)
        print('*model_body.input', model_body.input)

        return model_body
    else:
        raise Exception('Unknown model.')

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None, seg_shape=None):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        seg_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box, seg = get_random_data(annotation_lines[i], input_shape, random=True, seg_shape=seg_shape)
            image_data.append(image)
            box_data.append(box)
            seg_data.append(seg)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_seg_data = np.array(seg_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        #np.zeros(batch_size) -> seems like the default implementation send a dummy output.
        if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
            # y_true.append(y_seg_data)
            # yield ({'input_1': x1, 'input_2': x2}, {'output': y}) -> https://keras.io/models/model/
            yield ([image_data, *y_true],{'yolo_loss':np.zeros(batch_size), 'seg_output':y_seg_data})
        elif  model_name in ['tiny_yolo_seg']:
            yield ([image_data],{'seg_output':y_seg_data})
        else:
            yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None, seg_shape=None):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name, seg_shape=seg_shape)

if __name__ == '__main__':

    sentry_config = None
    with open("grv/sentry-config.yml", 'r') as stream:
        sentry_config = yaml.load(stream)
    sentry = Client(sentry_config['sentry-url'])

    train_config = None
    with open(ARGS.config_path, 'r') as stream:
        train_config = yaml.load(stream)
    print(train_config)

    output_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #infer_logdir_epochs_dataset_outputversion
    future_inference_outputpath = 'infer_{}_{}_{}_{}_{}'.format(
        train_config['log_dir'].replace('/',''),
        train_config['dataset_name'],
        train_config['model_name'],
        train_config['short_comment'] if train_config['short_comment'] else '',
        output_version,
        )
    os.makedirs(train_config['log_dir'], exist_ok=True)
    Path(os.path.join(train_config['log_dir'],future_inference_outputpath)).touch()

    # try:
    _main(train_config)
    # except Exception as e:
        # print('Captured Exception', e)
        # sentry.captureException()
