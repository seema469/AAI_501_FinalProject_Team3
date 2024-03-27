# -*- coding: utf-8 -*-
"""
Split the training image dataset in three single files. 
"""


import numpy as np
data = np.load('train_images_singleImages.npy')

split1, split2, split3 = np.array_split(data, 3)

np.save("train_split1.npy", split1)
np.save("train_split2.npy", split2)
np.save("train_split3.npy", split3)

# data1 = np.load('train_split1.npy')
# data2 = np.load('train_split2.npy')
# data3 = np.load('train_split3.npy')

# if (np.array_equal(split1, data1) and np.array_equal(split2, data2) and np.array_equal(split3, data3)):
#     print("EQUAL")
# else:
#     print("NOT EQUAL")