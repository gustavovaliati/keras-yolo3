import os
import glob
import argparse
import datetime
from tqdm import tqdm

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset_path",
                required=False,
                default=None,
                type=str,
                help="The dataset root path location.")
ap.add_argument("-t", "--txt_data_path",
                required=False,
                default=None,
                type=str,
                help=   "The txt file normally used in obj.data as train or valid."
                        " Good for when you need to create separated files.")
ap.add_argument("-e", "--min_height",
                required = False,
                default=0,
                type=int,
                help = "Discard any bbox bellow the min height.")

ARGS = ap.parse_args()

if not ARGS.dataset_path and not ARGS.txt_data_path:
    raise Exception('Missing parameter.')

if ARGS.dataset_path and ARGS.txt_data_path:
    raise Exception('Cannot inform both parameters. Choose only one.')

annotation_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_filename = None
img_list = None
if ARGS.dataset_path:
    img_list =  glob.glob(
        os.path.join(ARGS.dataset_path, '**/*.jpg'),
        recursive=True)
    output_filename = 'train_{}.txt'.format(annotation_version)
elif ARGS.txt_data_path:
    with open(ARGS.txt_data_path, 'r') as f:
        img_list = f.readlines()

    output_filename = '{}_{}.txt'.format(os.path.basename(ARGS.txt_data_path), annotation_version)
else:
    raise Exception('Missing parameters.')

if len(img_list) == 0:
    raise Exception('There are no annotations in the given dataset path.')

'''
Row format: image_file_path box1 box2 ... boxN;
Box format: x_min,y_min,x_max,y_max,class_id (no space).
'''
IMG_WIDTH = 640
IMG_HEIGHT = 480

discarded_by_insuficient_height = 0
with open(output_filename, 'w') as train_f:
    print('We have found {} images.'.format(len(img_list)))
    for img_file_path in tqdm(img_list):
        img_file_path = img_file_path.replace('\n','')
        annot_file_path = img_file_path.replace('.jpg', '.txt')
        train_f.write(img_file_path)
        with open(annot_file_path, 'r') as annot_f:
            for annot in annot_f:

                an = annot.split(' ')
                class_id = an[0]
                x_center = float(an[1])*(IMG_WIDTH - 1) #begins in zero
                y_center = float(an[2])*(IMG_HEIGHT - 1) #begins in zero
                w = float(an[3])*IMG_WIDTH #begins in one
                h = float(an[4])*IMG_HEIGHT #begins in one

                x_min = int(x_center - w/2)
                y_min = int(y_center - h/2)
                x_max = int(x_min + w)
                y_max = int(y_min + h)

                #handles annotation tool bug which permits annotating out of the image boundaries.
                if x_max >= IMG_WIDTH:
                    x_max = IMG_WIDTH - 1
                if y_max >= IMG_HEIGHT:
                    y_max = IMG_HEIGHT - 1

                if x_max == x_min or y_max == y_min:
                    print('Discarding:',img_file_path, x_min, y_min, x_max, y_max, class_id)
                    continue

                if y_max - y_min < ARGS.min_height:
                    discarded_by_insuficient_height += 1
                    continue

                train_f.write(' {},{},{},{},{}'.format(x_min, y_min, x_max, y_max, class_id))
        train_f.write('\n')

if discarded_by_insuficient_height > 0:
    print('{} bboxes have been discarded due insuficient height which minimum has been set to {}'.format(discarded_by_insuficient_height,ARGS.min_height))

print('The outputfile is', output_filename)
