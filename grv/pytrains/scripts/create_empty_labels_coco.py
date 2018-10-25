import argparse, os, glob, datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path",
    required = True,
    default = None,
    type=str,
    help = "COCO dataset path")
ARGS = ap.parse_args()

print('Using dataset path:', ARGS.dataset_path)

print('Reading dir...')
coco_imgs_list =  glob.glob(os.path.join(ARGS.dataset_path,'**/*.jpg'), recursive=True)
print('Done. # of JPG files found: {}. Processing...'.format(len(coco_imgs_list)))

log_file_path = '{}_{}.log'.format(os.path.basename(__file__), datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

touched = 0
with open(log_file_path, 'w') as log_f:
    for img in coco_imgs_list:
        label = img.replace('/images/','/labels/').replace('.jpg','.txt')
        if not os.path.exists(label):
            touched += 1
            log_f.write(label + '\n')
            Path(label).touch() #The touch function does not overwrite files.
print('Done. {} txt files have been created in the labels dir'.format(touched))
print('Check the log file for the list of the files created: ', log_file_path)
