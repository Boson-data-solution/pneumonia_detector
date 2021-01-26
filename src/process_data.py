import numpy as np
import cv2
import os
import pickle

import config as config


def get_data(data_dir, lables, img_size):
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

X_train, y_train = get_data('../data/train', config.LABELS, config.IMG_SIZE)
X_val, y_val = get_data('../data/val', config.LABELS, config.IMG_SIZE)
X_test, y_test = get_data('../data/test', config.LABELS, config.IMG_SIZE)

if not os.path.exists('../data/processed_data'):
    os.makedirs('../data/processed_data')

with open('../data/processed_data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('../data/processed_data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('../data/processed_data/X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)

with open('../data/processed_data/y_val.pkl', 'wb') as f:
    pickle.dump(y_val, f)

with open('../data/processed_data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('../data/processed_data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
