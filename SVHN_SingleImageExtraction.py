#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os


# In[9]:


def removeMultiDigitFilesAndCreateNPYarray(imageFilepath, outputFilename):
    
    CSVFilename = imageFilepath + "\\digitStruct.csv"
    # Define the rectangle coordinates (x, y, width, height)
    coordinates = pd.read_csv(CSVFilename)
    
    print("coordinates shape " , coordinates.shape)
    
    # Find all unique strings in the first column that occur more than once
    value_counts = coordinates['FileName'].value_counts()
    to_remove = value_counts[value_counts > 1].index.tolist()    
   
    # Remove rows where the first column matches any string in 'to_remove'
    filtered_coordinates = coordinates[~coordinates['FileName'].isin(to_remove)]    
    
    print(filtered_coordinates.head())
    print("filtered_coordinates shape " , filtered_coordinates.shape)
    
    # Get the rows where we have multiple digit files
    multi_digit_files = coordinates[coordinates['FileName'].isin(to_remove)]
    
    # Save multidigit file names and labels in this file
    np.save(outputFilename +'_multipleDigitPNG.npy', multi_digit_files)
    
    
    print("multi_digit_files shape " , multi_digit_files.shape)
    
    # Initialize an empty list to store image arrays
    image_arrays = []
    image_label_arrays = []
    
    # Loop through each file, load the image, and append its array to the list
    for index, row in filtered_coordinates.iterrows():
        if row['DigitLabel'] != 10:
            image_path = os.path.join(imageFilepath, row['FileName'])
            img = cv2.imread(image_path)
            # resized_image = cv2.resize(img, (32, 32))
            resized_image = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        
            image_array = np.array(resized_image)
            image_arrays.append(image_array)
            
            # image_label_array = np.array(row['DigitLabel'])
            image_label_arrays.append(row['DigitLabel'])
    
    # Stack all image arrays into a single NumPy array
    all_images_array = np.stack(image_arrays)
    
    # Save the NumPy array to a file
    np.save(outputFilename +'_images.npy', all_images_array)
    np.save(outputFilename + '_label.npy', image_label_arrays)
    print("single image len " , len(all_images_array))
    print("single image label len " , len(image_label_arrays))
    return outputFilename +'_multipleDigitPNG.npy'
   


# In[10]:


def segmentImagesAndSaveNPYarray(ImageFilePath, target):
    
    multi_images_NPY =  removeMultiDigitFilesAndCreateNPYarray(ImageFilePath, target)
    
    
    mult_image_files = np.load(multi_images_NPY, allow_pickle=True)
    image_list = []
    image_label_list = []
    for i in np.arange(len(mult_image_files)):
        try:
            # Load the original image
            image = cv2.imread(ImageFilePath + "\\" + mult_image_files[i,0])
                       
            # Define the rectangle coordinates (x, y, width, height)           
            coordinates = mult_image_files[i,2:]            
            x=coordinates[0]
            y=coordinates[1]
            w=coordinates[2]
            h=coordinates[3]        
    
            # Crop the image
            cropped_image = image[y:y+h, x:x+w]
            
            # Check if the image was loaded successfully
            if cropped_image is None:
                continue
            resized_image = cv2.resize(cropped_image, (32, 32))
            
            img_array = np.array(resized_image)
            image_list.append(img_array)
            image_label_list.append(mult_image_files[i,1])
        except cv2.error as e:
            print(f"Error resizing image at {mult_image_files[i,0]}: {e}. Skipping.")
            continue
       
    # Convert the List to a Numpy Array
    image_array = np.stack(image_list)  
    image_label_array = np.stack(image_label_list)  
    
    # Append single images 
    more_images = np.load(target + '_images.npy')
    more_labels = np.load(target + '_label.npy')
   
    image_array = np.concatenate((image_array, more_images), axis=0)
    image_label_array = np.concatenate((image_label_array, more_labels), axis=0)
    
    # Save the cropped images and labels
    np.save(target + "_images_singleImages.npy", image_array)
    np.save(target + "_labels_singleImages.npy", image_label_array)
    
    
    
    print("single image from mult-image len " , len(image_array))
    print("single image label from mult-image len  " , len(image_label_array))
 


def plotSegmentedImage(imageName):          
    # Load the image
    image = cv2.imread(imageName)    
    plt.imshow(image, cmap='gray')
    plt.show()


def testSavedImages(plotTest_images,plot_Test_labels,n,target):
    random_values = np.random.randint(0, 73171, size=n)
    _, axes = plt.subplots(nrows=1, ncols = n, figsize=(16,4))
    
    for i, ax in enumerate(axes.flatten()):
        # Use the random index to select an image
        img = plotTest_images[random_values[i]]
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Hide the axis
        ax.set_title("%i" % plot_Test_labels[random_values[i]])    
   


ImageFilePath = ".\\extra\\extra\\"

target = ".\\dataset\\extra"

segmentImagesAndSaveNPYarray(ImageFilePath, target)

test_images_ = np.load(target + '_images.npy')
test_labels_ = np.load(target + '_label.npy')
n = 10
_, axes = plt.subplots(nrows=1, ncols = 10, figsize=(16,4))
for ax, image, label in zip(axes, test_images_[0:n], test_labels_):
    ax.set_axis_off()
    ax.imshow(image, cmap='gray')
    
    ax.set_title(": %i" % label)



test_images = np.load(target+'_images_singleImages.npy')
test_labels = np.load(target+'_labels_singleImages.npy')
testSavedImages(test_images,test_labels,5,target)

