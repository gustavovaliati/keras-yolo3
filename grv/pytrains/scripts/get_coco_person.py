from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='..'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
# imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

#load and display image
imgpath = '%s/images/%s/%s'%(dataDir,dataType,img['file_name'])
labelspath = imgpath.replace('.jpg', '.txt')
print(os.path.abspath(imgpath))
I = io.imread(imgpath)

h,w,x = I.shape
print(I.shape)
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for ann in anns:
    rle = coco.annToRLE(ann)
    bboxes = maskUtils.toBbox(rle)
    print(bboxes)

## load and display instance annotations
# plt.imshow(I); plt.axis('off')
# coco.showAnns(anns)
# plt.show()
