"""
Retrain the YOLO model for your own dataset.
"""

import argparse
import yaml
from raven import Client

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

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, tiny_yolo_infusion_body
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
    freeze_body = 0
    weights_path = None

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=False,
            freeze_body=freeze_body, weights_path=weights_path, model_name=model_name)
    else:
        model = create_model(input_shape, anchors, num_classes, load_pretrained=False,
            freeze_body=freeze_body, weights_path=weights_path, model_name=model_name) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
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

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    # if False:
    #     model.compile(optimizer=Adam(lr=1e-3), loss={
    #         # use custom yolo_loss Lambda layer.
    #         'yolo_loss': lambda y_true, y_pred: y_pred})
    #
    #     batch_size = 32
    #     # batch_size = 1
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    #             steps_per_epoch=max(1, num_train//batch_size),
    #             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    #             validation_steps=max(1, num_val//batch_size),
    #             epochs=1,
    #             initial_epoch=0,
    #             callbacks=[logging, checkpoint])
    #     model.save_weights(log_dir + 'trained_weights_stage_1.h5')
    #
    # # Unfreeze and continue training, to fine-tune.
    # # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred,
            }) # recompile to apply the change
        # print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, model_name),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, model_name),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


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

    if model_name == 'tiny_yolo_infusion':
        y_true_input = [
            Input(
                shape=( h//{0:32, 1:16}[l],
                        w//{0:32, 1:16}[l],
                        num_anchors//2,
                        num_classes+5 )
                ) for l in range(2)
            ]
        y_true_input.append(Input(shape=(None, None, 2)))#add segmentation y input.

        model_body = tiny_yolo_infusion_body(image_input, num_anchors//2, num_classes)
        print('Create Tiny YOLOv3 INFUSION model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            print("####### WARNING ####### Review whether the parameters for freezing are OK.")
            raise Exception('freezing requires review.')
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        '''
            def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
                ...
                return loss
        '''
        print('Checking Lambda', [*model_body.output, *y_true_input])
        '''
        Checking Lambda [
        <tf.Tensor 'yolo_head_a_output/BiasAdd:0' shape=(?, ?, ?, 18) dtype=float32>,
        <tf.Tensor 'yolo_head_b_output/BiasAdd:0' shape=(?, ?, ?, 18) dtype=float32>,
        <tf.Tensor 'seg_output/LeakyRelu/Maximum:0' shape=(?, ?, ?, 2) dtype=float32>,
        <tf.Tensor 'input_2:0' shape=(?, 13, 13, 3, 6) dtype=float32>,
        <tf.Tensor 'input_3:0' shape=(?, 26, 26, 3, 6) dtype=float32>,
        <tf.Tensor 'input_4:0' shape=(?, ?, ?, 2) dtype=float32>]
        '''


        model_loss = Lambda(
                        yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={
                            'anchors': anchors,
                            'num_classes': num_classes,
                            'ignore_thresh': 0.7,
                            'model_name': model_name,
                            'print_loss': False
                        }
                    )([*model_body.output, *y_true_input])#this is calling yolo_loss and these are the args.
                    # model_body.output is the last layer output tensor.

        model = Model([model_body.input, *y_true_input], model_loss)

        return model
    else:
        y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]

        model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(
                        yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={
                            'anchors': anchors,
                            'num_classes': num_classes,
                            'ignore_thresh': 0.7
                        }
                    )([*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None):
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
            image, box, seg = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            seg_data.append(seg)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_seg_data = np.array(seg_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
            y_true.append(y_seg_data)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name)

if __name__ == '__main__':

    sentry_config = None
    with open("grv/sentry-config.yml", 'r') as stream:
        sentry_config = yaml.load(stream)
    sentry = Client(sentry_config['sentry-url'])

    train_config = None
    with open(ARGS.config_path, 'r') as stream:
        train_config = yaml.load(stream)
    print(train_config)
    # try:
    _main(train_config)
    # except Exception as e:
        # print('Captured Exception', e)
        # sentry.captureException()
