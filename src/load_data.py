import numpy as np
import cv2
import os

    
def load_data(data_dir, lables, img_size):
    data = [] 
    for label in lables: 
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            if 'virus' in img:
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, 1])
                except Exception as e:
                    print(e)
            elif 'bacteria' in img:
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, 2])
                except Exception as e:
                    print(e)
            else:
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, 0])
                except Exception as e:
                    print(e)
    labelled_data = np.array(data)
    X = []
    y = []
    for feature, label in labelled_data:
        X.append(feature)
        y.append(label)
    return X, y
