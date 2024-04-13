# SVHN Single Digit Classification

## Introduction

The objective of this project is to create a robust model that can effectively recognize single digits (0-9) in images taken from Google Street View. 

# Package Installation

Install following packages using pip install command-
pip install numpy 
pip install matplotlib
pip install tensorflow
pip install opencv-python
pip install scikit-learn



## Dataset

The SVHN dataset (http://ufldl.stanford.edu/housenumbers/), available in its raw form, comprises full-color images of house numbers captured from Google Street View. These images, initially in .png format, were downloaded from the official SVHN dataset repository. Accompanying the images, metadata files containing the coordinates of digit bounding boxes were also downloaded. These files are essential for the subsequent extraction of single-digit images from the larger, multi-digit compositions.The images of house number features a sequence of digits within a single image. Rather than recognizing the entire sequence of digits simultaneously, we tackle the task by cropping the image into separate segments. Each segment contains an individual digit, which we then classify independently. To effectively isolate these individual digits for classification, we have implemented a series of preprocessing steps.



## Pre-processing

Extraction of Single-Digit Images and labeling involved processing the images and their corresponding metadata to isolate individual digits. The metadata files provided precise coordinates for the bounding boxes surrounding each digit within the multi-digit images. Utilizing these coordinates, we used a script (SVHN_SingleImageExtraction.py) to crop the original images accordingly, thereby extracting single-digit images and resizing them to a uniform dimension of 32x32x3 pixels. 
This repository from GitHub is used to get the matfile to csv conversion : https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader.git.

## Custom CNN Implementation

SVHN_ImageClassification_CNN.ipynb implements the custom CNN model and provides the execution results and plots.

## VGG16 Transfer Learning

SVHN_ImageClassification_VGG16.ipynb implements the transfer learning functionality and provides the execution results and plots.

```python

```
