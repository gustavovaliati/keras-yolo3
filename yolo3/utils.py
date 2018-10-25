"""Miscellaneous utility functions."""

from functools import reduce
import math
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import hashlib
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size, black_white=False):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    if black_white:
        new_image = Image.new('L', size, (0))
    else:
        new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def update_annotation(old_annotation_line, box_data):
    new_annot = old_annotation_line.split(' ')[0]

    for an in box_data:
        if np.sum(an) > 0:
            an = list(map(int, an))
            new_annot += ' {},{},{},{},{}'.format(an[0],an[1],an[2],an[3],an[4])

    return new_annot

def get_random_data(annotation_line, input_shape, random='full', max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, model_name=None, use_seg_mask_generator=False):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    # print('ih,iw',ih,iw)
    # print('seg_shape',seg_shape)
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        '''
        There is no need to update the annotation_line when the image is just resized.
        The get_seg_data already handles that.
        '''
        seg_data = None
        if model_name in ['tiny_yolo_infusion', 'yolo_infusion']:
            seg_data = get_seg_data(annotation_line, img_shape=(ih,iw), input_shape=input_shape, model_name=model_name, use_seg_mask_generator=use_seg_mask_generator)

        return image_data, box_data, seg_data

    if random in ['flip_only']:
        flip = rand()<.5

        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            if flip: new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy

            #fix flip
            if flip: box[:, [0,2]] = w - box[:, [2,0]]

            box_data[:len(box)] = box



        '''
        There is no need to update the annotation_line when the image is just resized.
        The get_seg_data already handles that.
        '''
        seg_data = None
        if model_name in ['tiny_yolo_infusion','yolo_infusion']:
            seg_data = get_seg_data(annotation_line, img_shape=(ih,iw), input_shape=input_shape, model_name=model_name, use_seg_mask_generator=use_seg_mask_generator)

        return image_data, box_data, seg_data

    if random == 'full':
        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        seg_data = None
        if model_name in ['tiny_yolo_infusion','yolo_infusion']:
            updated_annotation_line = update_annotation(annotation_line, box_data)
            n_iw, n_ih = image.size
            seg_data = get_seg_data(updated_annotation_line, img_shape=(n_ih,n_iw), input_shape=input_shape, model_name=model_name, use_seg_mask_generator=use_seg_mask_generator)
        # plt.imshow(image_data)
        # plt.savefig('image_data.jpg')
        # plt.imshow(seg_data[:,:,0],cmap='jet')
        # plt.savefig('fg_mask.jpg')
        # plt.imshow(seg_data[:,:,1],cmap='jet')
        # plt.savefig('bg_mask.jpg')
        # print(annotation_line)
        # print(updated_annotation_line)
        # print(seg_data[:,:,0])
        # import sys
        # sys.exit()

        return image_data, box_data, seg_data
    else:
        raise Exception('The specified data augmentation method is not recognized:', random)

def old_get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, seg_shape=None):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    # print('ih,iw',ih,iw)
    # print('seg_shape',seg_shape)
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    seg_data = get_seg_data(annotation_line, img_shape=(ih,iw), input_shape=seg_shape)

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data, seg_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data, seg_data

def seg_mask_generator(annotation_line, img_shape, wanted_shape):
    '''
    image_shape = (h,w)
    wanted_shape = (h,w)
    '''
    annot = annotation_line.split(' ')
    i_h, i_w = img_shape
    n_h, n_w = wanted_shape
    fg_mask = np.zeros((wanted_shape), dtype=np.uint8)

    for bbox in annot[1:]:
        x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
        # print('x_min, y_min, x_max, y_max',x_min, y_min, x_max, y_max)
        n_x_min = int(n_w * x_min / i_w)
        n_x_max = math.ceil(n_w * x_max / i_w) # want to guarantee at least one pixel always

        n_y_min = int(n_h * y_min / i_h)
        n_y_max = math.ceil(n_h * y_max / i_h) # want to guarantee at least one pixel always

        fg_mask[n_y_min:n_y_max, n_x_min:n_x_max] = 1
        # print('n_x_min, n_y_min, n_x_max, n_y_max',n_x_min, n_y_min, n_x_max, n_y_max)

    bg_mask = np.logical_not(fg_mask).astype(int)
    return np.dstack((fg_mask, bg_mask))

def get_seg_data(annotation_line, img_shape, input_shape, model_name=None, use_seg_mask_generator=True):
    ''' Returns the y_true matrix for the weak seg head for a specific annot image
        Parameters
        ----------
        annotation_line: string, default annotation unit
        img_shape: array-like, hw, original image dimensions
        input_shape: array-like, hw, network input shape

        Returns
        -------
        array: (h,w,2), 2 layers: one for foreground and the other for
            background information about the annotated segmentated object.
    '''


    ih, iw = img_shape
    input_h, input_w = input_shape

    if model_name in ['tiny_yolo_infusion', 'yolo_infusion', 'tiny_yolo_seg']:

        if input_h % 32 == 0 and input_w % 32 == 0:
            # trying to infer automaticaly
            seg_h, seg_w = input_h // 32, input_w // 32
        else:
            raise Exception('Unknown seg configuration')

        # print('get_seg_data ih,iw ',ih,iw )


        if use_seg_mask_generator:
            return seg_mask_generator(annotation_line, img_shape=img_shape, wanted_shape=(seg_h,seg_w))
        else:

            fg_mask = np.zeros((ih,iw), dtype=np.uint8)
            annot = annotation_line.split(' ')
            img_path = annot[0]
            for bbox in annot[1:]:
                x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
                fg_mask[y_min:y_max, x_min:x_max] = 1

            fg_mask = np.array(letterbox_image(Image.fromarray(fg_mask,'L'), (seg_w,seg_h), black_white=True))
            bg_mask = np.logical_not(fg_mask).astype(int)

            # print(annotation_line)
            # print(fg_mask)
            # print(bg_mask)
            # plt.imshow(fg_mask, cmap='jet')
            # plt.savefig('seg_output.jpg')
            # import sys
            # sys.exit()

            return np.dstack((fg_mask, bg_mask))

    elif model_name in ['tiny_yolo_infusion_hydra']:
        hydra_config = 1
        if hydra_config == 1:
            if input_h == 416 and input_w == 416:
                seg_heads_inputs = [(26, 26),(13, 13)] # [(h,w), (h,w)]
            elif input_h == 480 and input_w == 640:
                seg_heads_inputs = [(30, 40),(15, 20)] # [(h,w), (h,w)]
            else:
                raise Exception('Unknown hydra inputs configuration')
        else:
            raise Exception('Unknown hydra configuration')

        fg_mask = np.zeros((ih,iw), dtype=np.uint8)
        annot = annotation_line.split(' ')
        img_path = annot[0]
        for bbox in annot[1:]:
            x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
            fg_mask[y_min:y_max, x_min:x_max] = 1

        seg_heads_masks = []
        for seg_heads_input in seg_heads_inputs:
            seg_h,seg_w = seg_heads_input
            fg_mask = np.array(letterbox_image(Image.fromarray(fg_mask,'L'), (seg_w,seg_h), black_white=True))
            bg_mask = np.logical_not(fg_mask).astype(int)
            seg_heads_masks.append(np.dstack((fg_mask, bg_mask)))
        return seg_heads_masks

    else:
        raise Exception('Unknown model name')

def build_class_translation_map(classes, translation_config):
    print('Building class translation...')
    print('Original[id] -> New[id]')
    mapping = []
    for class_id, class_name in enumerate(classes):
        if class_name not in translation_config:
            raise Exception('Class translation: missing translation config for ', class_name)
        new_class_name = translation_config[class_name]
        if new_class_name == 'discard':
            mapping.append(-1)
            print('{}[id {}] -> discarded'.format(class_name,class_id))
        else:
            if new_class_name not in classes:
                raise Exception('Class translation: class name in config is not present in the classes file.')
            new_class_id = int(classes.index(new_class_name))
            print('{}[id {}] -> {}[id {}]'.format(class_name,class_id,new_class_name,new_class_id))
            # in the array of position class_id there will be the new_class_id.
            mapping.append(new_class_id)
    return mapping


def translate_classes(lines, classes, translation_config):
    new_lines= []
    class_mapping = build_class_translation_map(classes, translation_config)
    for annot_line in lines:
        new_annot_line = ''
        splitted = annot_line.split(' ')
        img_path = splitted[0].strip()
        new_annot_line += img_path
        for bbox in splitted[1:]:
            x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
            new_class_id = class_mapping[int(class_id)]
            if new_class_id >= 0: #check if we need to discard this bbox according to translation config.
                new_annot_line += ' {},{},{},{},{}'.format(x_min, y_min, x_max, y_max, new_class_id)
        new_lines.append(new_annot_line)
    return new_lines

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

def calc_annot_lines_md5(lines_array):
    hash = hashlib.md5()
    for line in lines_array:
        hash.update(line.strip().encode('utf-8'))
    return hash.hexdigest()
