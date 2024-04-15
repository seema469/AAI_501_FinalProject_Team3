# SVHN Single Digit Classification Using Real-world Images

## Introduction

The objective of this project is to create a robust model that can effectively recognize single digits (0-9) in images taken from Google Street View. 

## Package Installation

Install following packages using pip install command-
pip install numpy 
pip install matplotlib
pip install tensorflow
pip install opencv-python
pip install scikit-learn

## Dataset

The SVHN dataset (http://ufldl.stanford.edu/housenumbers/), available in its raw form, comprises full-color images of house numbers captured from Google Street View. These images, initially in .png format, were downloaded from the official SVHN dataset repository. Accompanying the images, metadata files containing the coordinates of digit bounding boxes were also downloaded. These files are essential for the subsequent extraction of single-digit images from the larger, multi-digit compositions.The images of house number features a sequence of digits within a single image. Rather than recognizing the entire sequence of digits simultaneously, we tackle the task by cropping the image into separate segments. Each segment contains an individual digit, which we then classify independently.

## Pre-processing

Extraction of Single-Digit Images and labeling involved processing the images and their corresponding metadata to isolate individual digits. The metadata files provided precise coordinates for the bounding boxes surrounding each digit within the multi-digit images. Utilizing these coordinates, we used a script (SVHN_SingleImageExtraction.py) to crop the original images accordingly, thereby extracting single-digit images and resizing them to a uniform dimension of 32x32x3 pixels. 
This repository from GitHub is used to get the matfile to csv conversion : https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader.git.

## Implementation

### SVHN_SingleImageExtraction.py 

This script will extract the single digit images and store them as a numpy (.npy) file.

For example to create train_images_singleImages.npy and train_labels_singleImages.npy set following
    ImageFilePath = ".\\extra\\extra\\"
    target = ".\\dataset\\train"

.\\dataset is where all the .npy files will be stored.

ImageFilePath - .png files are stored in this folder path. For example training dataset path could be ..\\train\\train\\"

target - the train_images_singleImages.npy and train_labels_singleImages.npy will be stored at location ".\\dataset\\"



### Custom CNN Implementation

SVHN_ImageClassification_CNN.ipynb implements the custom CNN model and provides the execution results and plots.

### VGG16 Transfer Learning

SVHN_ImageClassification_VGG16.ipynb implements the transfer learning functionality and provides the execution results and plots.

### CombineDataSet

This Python script is used to combine split files. Unzips each split and modify the script code based on the number of splits that need to be combined.

### SplitDataSet 

This Python script is used to split large NumPy (.npy) files. This is necessary due to GitHub's file size limit for check-ins. After splitting, each segment can be zipped. The zipped size of each segment should not exceed GitHub's file size limitation.

