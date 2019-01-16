"""
Retrain the YOLO model for your own dataset.
"""

import argparse
import yaml
from raven import Client
import datetime
import os
from pathlib import Path
import re

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
ap.add_argument("-w", "--pretrained_weights",
                required=False,
                default=None,
                type=str,
                help="The weights to pre-load in the model. This has priority over the config file.")
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

from yolo3.model import (preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss,
    tiny_yolo_infusion_body, infusion_layer, yolo_infusion_body, tiny_yolo_infusion_hydra_body,
    yolo_body_for_small_objs, tiny_yolo_small_objs_body)
from yolo3.utils import get_random_data, get_classes, translate_classes, calc_annot_lines_md5

def _main(train_config):
    annotation_path = train_config['train_path']
    log_dir = train_config['log_dir']
    classes_path = train_config['classes_path']
    anchors_path = train_config['anchors_path']
    model_name = train_config['model_name']
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    print('num_classes', num_classes)
    num_yolo_heads = 3 if model_name in ['yolo', 'yolo_infusion'] else 2 #This is the number of specifically YOLO heads which predict bboxes. Not counting infusion, etc.
    print('number of yolo heads', num_yolo_heads)
    anchors = get_anchors(anchors_path, num_yolo_heads)
    freeze_body = 1
    pretrained_weights_path = ARGS.pretrained_weights if ARGS.pretrained_weights else train_config['pretrained_weights_path']
    input_shape = (int(train_config['input_height']),int(train_config['input_width']))

    if model_name in ['tiny_yolo', 'tiny_yolo_infusion']:
        model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True,
            freeze_body=freeze_body, weights_path=pretrained_weights_path, model_name=model_name, num_yolo_heads=num_yolo_heads)
    else:
        model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
            freeze_body=freeze_body, weights_path=pretrained_weights_path, model_name=model_name, num_yolo_heads=num_yolo_heads) # make sure you know what you freeze

    # print(model.summary())

    logging = TensorBoard(log_dir=log_dir, write_grads=True, write_images=True)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()

    if 'class_translation_path' in train_config and train_config['class_translation_path']:
        print('Translating dataset classes...')
        with open(train_config['class_translation_path'], 'r') as stream:
            class_translation_config = yaml.load(stream)

        lines = translate_classes(lines,class_names,class_translation_config)
        print('Translation is done. Now we want to save the new translated dataset version.')
        annotation_path_translated = annotation_path.replace('.txt', '_'+train_config['class_translation_path'].replace('.yml', '.txt'))
        if os.path.exists(annotation_path_translated):
            print('Seems like this translation has already been done before.')
            already_present_translation_on_disk_lines = open(annotation_path_translated, 'r').readlines()
            disk_md5 = calc_annot_lines_md5(already_present_translation_on_disk_lines)
            disk_md52 = calc_annot_lines_md5(already_present_translation_on_disk_lines)
            current_translated_md5 = calc_annot_lines_md5(lines)
            print('Checking translation version...')
            if disk_md5 == current_translated_md5:
                print('Disk translation version matches the current generated one. Lets procced to training.')
            else:
                print('Disk translation version is different from the current one. Seems like the translation code has changed.')
                print('Do backup the disk annotation translated file and properly document it, and move to some other folder: ', annotation_path_translated)
                raise Exception('Disk and current class translations missmatch versions. Cannot proceed.')
        else:
            with open(annotation_path_translated, 'w') as output_f:
                print('Writting the new translated annotation file to', annotation_path_translated)
                for annot_line in lines:
                    output_f.write(annot_line + '\n')

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    batch_size_freezed = train_config['batch_size_freezed']
    epochs_freezed = train_config['epochs_freezed']
    if True and epochs_freezed > 0:
        compile_model(model, model_name)
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_freezed))
        model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size_freezed, input_shape, anchors, num_classes, model_name, num_yolo_heads),
                steps_per_epoch=max(1, num_train//batch_size_freezed),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size_freezed, input_shape, anchors, num_classes, model_name, num_yolo_heads),
                validation_steps=max(1, num_val//batch_size_freezed),
                epochs=epochs_freezed,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir,'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        #Unfreeze all layers.
        print('Unfreeze all of the layers.')
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        #recompile the model once we unfreezed the layers.
        compile_model(model, model_name)


        batch_size_unfreezed = train_config['batch_size_unfreezed'] # note that more GPU memory is required after unfreezing the body
        epochs_unfreezed = train_config['epochs_unfreezed']
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfreezed))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size_unfreezed, input_shape, anchors, num_classes, model_name, num_yolo_heads),
            steps_per_epoch=max(1, num_train//batch_size_unfreezed),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size_unfreezed, input_shape, anchors, num_classes, model_name, num_yolo_heads),
            validation_steps=max(1, num_val//batch_size_unfreezed),
            epochs=epochs_freezed + epochs_unfreezed,
            initial_epoch=epochs_freezed,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(os.path.join(log_dir,'trained_weights_final.h5'))

    # Further training if needed.


def compile_model(model, model_name):
    print('model_name',model_name)
    learning_rate = train_config['initial_lr'] if 'initial_lr' in train_config else 1e-4

    #old style
    # model.compile(
    #     optimizer=Adam(lr=1e-4),
    #     loss={
    #         'yolo_loss': lambda y_true, y_pred: y_pred,
    #     }) # recompile to apply the change
    # return

    #new style
    if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
        model.compile(
            optimizer=Adam(lr=learning_rate),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred, #I guess this is a dumb operation. Does nothing
                'seg_output' : 'categorical_crossentropy'
                # 'seg_output' : 'binary_crossentropy'
            },
            loss_weights={'yolo_loss': 1., 'seg_output': train_config['seg_loss_weight']} #updating yolo_loss may not affect.
            )
    elif model_name in ['tiny_yolo_infusion_hydra']:
        model.compile(
            optimizer=Adam(lr=learning_rate),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred, #I guess this is a dump operation. Does nothing
            },
            loss_weights={'yolo_loss': 1.0, 'seg_output_1': 2.0, 'seg_output_2': 2.0} #updating yolo_loss may not affect.
            )
    elif model_name in ['tiny_yolo', 'yolo']:
        model.compile(
            optimizer=Adam(lr=learning_rate),
            loss={
                'yolo_loss': lambda y_true, y_pred: y_pred,
            }) # recompile to apply the change
    else:
        raise Exception('The model_name is unknown: ', model_name)

def get_anchors(anchors_path, num_yolo_heads):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    if len(anchors) % 2 != 0:
        raise Exception('The anchors should be in pairs.')
    anchors = np.array(anchors).reshape(-1, 2)
    if len(anchors) % num_yolo_heads != 0:
        raise Exception('The number of anchors is incompatible to the number of heads. Should be multiple of {}'.format(num_yolo_heads))
    return anchors


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', model_name=None, num_yolo_heads=None):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    if model_name == 'yolo_infusion':
        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, num_classes+5)) for l in range(3)]

        model_body, seg_output = yolo_infusion_body(image_input, num_anchors//3, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers)-3)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name='yolo_loss',
            arguments={
                'anchors': anchors,
                'num_classes': num_classes,
                'ignore_thresh': 0.5,
                'model_name': model_name,
                'num_yolo_heads': num_yolo_heads
            })([*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], outputs=[model_loss, seg_output])
        # print(model.summary())
        return model
    elif model_name in ['yolo', 'yolo_small_objs']:

        if model_name == 'yolo_small_objs':
            y_true = [Input(shape=(h//{0:32, 1:16, 2:4}[l], w//{0:32, 1:16, 2:4}[l], \
                num_anchors//3, num_classes+5)) for l in range(3)]
            model_body = yolo_body_for_small_objs(image_input, num_anchors//3, num_classes)
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
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'num_yolo_heads': num_yolo_heads})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5', model_name=None, num_yolo_heads=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    if model_name in ['tiny_yolo_infusion', 'tiny_yolo_infusion_hydra']:
        y_true_input = [
            Input(
                shape=( h//{0:32, 1:16}[l],
                        w//{0:32, 1:16}[l],
                        num_anchors//2,
                        num_classes+5 )
                ) for l in range(2)
            ]


        #old style
        if model_name == 'tiny_yolo_infusion_hydra':
            #old style
            # y_true_input += [Input(shape=(None, None, 2)), Input(shape=(None, None, 2))]
            # model_body = tiny_yolo_infusion_hydra_body(image_input, num_anchors//2, num_classes)
            #new style
            #keep commented.
            model_body, connection_layer = tiny_yolo_infusion_hydra_body(image_input, num_anchors//2, num_classes)
        elif model_name == 'tiny_yolo_infusion':
            #old style
            # y_true_input.append(Input(shape=(None, None, 2)))#add segmentation y input.
            #model_body = tiny_yolo_infusion_body(image_input, num_anchors//2, num_classes)
            #new style
            model_body, connection_layer = tiny_yolo_infusion_body(image_input, num_anchors//2, num_classes)
        #new style: keep commented.


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

        '''
            def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
                ...
                return loss
        '''
        print('*model_body.output', *model_body.output)
        print('*model_body.input', model_body.input)
        print('*y_true_input', *y_true_input)
        '''
        Checking Lambda [
        <tf.Tensor 'yolo_head_a_output/BiasAdd:0' shape=(?, ?, ?, 18) dtype=float32>,
        <tf.Tensor 'yolo_head_b_output/BiasAdd:0' shape=(?, ?, ?, 18) dtype=float32>,
        <tf.Tensor 'seg_output/LeakyRelu/Maximum:0' shape=(?, ?, ?, 2) dtype=float32>,
        <tf.Tensor 'input_2:0' shape=(?, 13, 13, 3, 6) dtype=float32>,
        <tf.Tensor 'input_3:0' shape=(?, 26, 26, 3, 6) dtype=float32>,
        <tf.Tensor 'input_4:0' shape=(?, ?, ?, 2) dtype=float32>]
        '''


        default_output = Lambda(
                        yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={
                            'anchors': anchors,
                            'num_classes': num_classes,
                            'ignore_thresh': 0.7,
                            'model_name': model_name,
                            'num_yolo_heads': num_yolo_heads,
                            'print_loss': False
                        }
                    )([*model_body.output, *y_true_input])#this is calling yolo_loss and these are the args.
                    # model_body.output is the last layer output tensor.

        #old style
        # model = Model([model_body.input, *y_true_input], default_output)
        '''
            model_body.input = image_input = Input(shape=(None, None, 3))
            *y_true_input = input_y_true_layer1, input_y_true_layer2, ...
            default_output = yolo_loss
        '''
        #new style
        seg_output = infusion_layer(connection_layer)
        #[model_body.input, *y_true_input] -> [images, y_layer_1, y_layer_2]
        model = Model([model_body.input, *y_true_input], outputs=[default_output, seg_output])

        return model

    elif model_name in ['tiny_yolo', 'tiny_yolo_small_objs']:
        if model_name == 'tiny_yolo_small_objs':
            y_true = [Input(shape=(h//{0:32, 1:8}[l], w//{0:32, 1:8}[l], \
                num_anchors//2, num_classes+5)) for l in range(2)]
            model_body = tiny_yolo_small_objs_body(image_input, num_anchors//2, num_classes)
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
                            'ignore_thresh': 0.7,
                            'num_yolo_heads': num_yolo_heads,
                            'model_name' : model_name
                        }
                    )([*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model
    else:
        raise Exception('unknown model.')

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None, num_yolo_heads=None):
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
            image, box, seg = get_random_data(annotation_lines[i], input_shape, random=train_config['data_augmentation'], model_name=model_name)
            image_data.append(image)
            box_data.append(box)
            seg_data.append(seg)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_seg_data = np.array(seg_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, model_name=model_name, num_yolo_heads=num_yolo_heads)

        if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
            #old style
            # yield [image_data, *y_true, y_seg_data], np.zeros(batch_size)

            #new style
            # yield ({'input_1': x1, 'input_2': x2}, {'output': y}) -> https://keras.io/models/model/
            yield ([image_data, *y_true],{'yolo_loss':np.zeros(batch_size), 'seg_output':y_seg_data})
        elif model_name in ['tiny_yolo_infusion_hydra']:
            #old style
            # y_seg_reorg = list(zip(*y_seg_data))
            # y_seg_data_1 = np.array(y_seg_reorg[0])
            ## print('y_seg_data_1',y_seg_data_1.shape)
            # y_seg_data_2 = np.array(y_seg_reorg[1])
            ## print('y_seg_data_2',y_seg_data_2.shape)
            # yield [image_data, *y_true, y_seg_data_1, y_seg_data_2 ], np.zeros(batch_size)
            #new style: not implemented yet.
            pass

        elif model_name in ['tiny_yolo','yolo', 'yolo_small_objs', 'tiny_yolo_small_objs']:
            yield [image_data, *y_true], np.zeros(batch_size) #np.zeros(batch_size) -> seems like the default implementation send a dummy output.
        else:
            raise Exception('unknown model.')

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name=None, num_yolo_heads=None):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, model_name, num_yolo_heads=num_yolo_heads)

if __name__ == '__main__':

    sentry_config = None
    with open("grv/sentry-config.yml", 'r') as stream:
        sentry_config = yaml.load(stream)
    sentry = Client(sentry_config['sentry-url'])

    train_config = None
    with open(ARGS.config_path, 'r') as stream:
        train_config = yaml.load(stream)
    print(train_config)

    experiment_number_regex = re.compile('[0-9]{3}.yml')
    m = experiment_number_regex.search(os.path.basename(ARGS.config_path))
    if m:
        experiment_number_str = m.group().replace('.yml','')
    else:
        raise Exception("Could not find the experiment number in the config file name.")

    if not train_config['log_dir'].endswith(experiment_number_str):
        raise Exception("The experiment number from the log_dir in the config file does not match the config file name.")

    output_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #infer_logdir_epochs_dataset_outputversion
    future_inference_outputpath = 'infer_{}_{}_{}_batch{}_{}-freezed_{}_{}'.format(
        train_config['log_dir'].replace('/',''),
        train_config['dataset_name'],
        train_config['model_name'],
        train_config['batch_size_freezed'],
        train_config['epochs_freezed'],
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
