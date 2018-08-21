# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

This project is a fork from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

**Warning**: This fork has not been exhaustively tested and bugs are expected.


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).

```
wget https://pjreddie.com/media/files/yolov3.weights
```

2. Convert the Darknet YOLO model to a Keras model.
3. Prepare your project config file in `yml`

```
train_path: my_train_set.txt
test_path: my_test_set.txt
classes_path: model_data/my_classes.txt
anchors_path: model_data/my_anchors.txt
model_name: any_name_for_my_model
log_dir: logs/my-test/
```

4. Training

`python3 train.py ---config_path myprojects/test1-config.yml -m 5000`

Ps: For training the `-m` parameter is available to set a limit to GPU memory in MB.

5. Inference

`python3 yolo.py ---config_path myprojects/test1-config.yml --weights logs/seg-000/ep004-loss-106.913-val_loss-114.463.h5`

Ps: For inference the memory is limited in 30%.

6. MultiGPU usage is an optional. Change the number of gpu and add gpu device id

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

1.1 To generate the train.txt file from a pre-configured folder used in
darknet training, you can use the darknet_annotation.py script and change
the WIDTH and HEIGHT parameters.

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights in yolo.py.  
    Remember to modify class path or anchor path.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

## Prediction

The `yolo.py` script has been modified and now outputs the `inference_output_<version>.txt` file every time it is ran. The `<version>` is the current datetime and exists to create a simple inference history.

1. Prediction output format.  
    One row for one image;  
    Row format: `image_file_path prediction1 predicion2 ... predictionN`;  
    Prediction format: `x_min,y_min,x_max,y_max,class_id,confidence_score` (no space).  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0,0.9876 30,50,200,120,3,0.3211
    path/to/img2.jpg
    path/to/img3.jpg 120,300,250,600,2,0.8319
    ...
    ```
---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
