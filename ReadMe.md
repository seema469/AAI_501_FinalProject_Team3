# SVHN Single Digit Classification

## Introduction

The objective of this project is to create a robust model that can effectively recognize single digits (0-9) in images taken from Google Street View. These images come with unique challenges, such as variability in digit appearance, background clutter, and image quality, making the task notably demanding.

## Dataset

The SVHN dataset (http://ufldl.stanford.edu/housenumbers/), available in its raw form, comprises full-color images of house numbers captured from Google Street View. These images, initially in .png format, were downloaded from the official SVHN dataset repository. Accompanying the images, metadata files containing the coordinates of digit bounding boxes were also downloaded. These files are essential for the subsequent extraction of single-digit images from the larger, multi-digit compositions. s


## Pre-processing

Extraction of Single-Digit Images and labeling involved processing the images and their corresponding metadata to isolate individual digits. The metadata files provided precise coordinates for the bounding boxes surrounding each digit within the multi-digit images. Utilizing these coordinates, we used a script to crop the original images accordingly, thereby extracting single-digit images and resizing them to a uniform dimension of 32x32x3 pixels. Each extracted single-digit image was then labeled based on the digit it represented, as indicated in the metadata files.
Normalization and One-hot Encoding
Following the organization of the extracted and labeled images, we proceeded with the normalization of pixel values. Each pixel in the RGB images was normalized by dividing by 255, scaling the values to a [0,1] range. This normalization aids in the model's training efficiency and convergence.
Additionally, the numerical labels were transformed into a one-hot encoded format, converting each label into a binary vector of length equal to the number of classes (10). This transformation is pivotal for aligning the labels with the softmax output layer of our CNN model, enabling a straightforward evaluation of model accuracy and loss during training.
This repository from GitHub is used to get the matfile to csv conversion : https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader.git.


```python

```
