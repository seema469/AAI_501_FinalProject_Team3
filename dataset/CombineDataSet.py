# -*- coding: utf-8 -*-
"""
Combine the training image files in a single file. 
"""

import numpy as np
split1 = np.load('train_split1.npy')
split2 = np.load('train_split2.npy')
split3 = np.load('train_split3.npy')

combined_data = np.concatenate((split1, split2, split3))

np.save("train_images_singleImages.npy", combined_data)
