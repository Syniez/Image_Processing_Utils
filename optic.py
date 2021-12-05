import numpy as np
import sklearn
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import MaxPool2D


im1_path = "./Blender/data/old_old_class/cam20001.jpg"
im2_path = "./Blender/data/old_old_class/cam20020.jpg"


def flow(im1, im2, feature_num):
    feature_params = dict(maxCorners = feature_num, qualityLevel = 0.01, minDistance = 7, blockSize = 7 ) 
    lk_params = dict( winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.zeros([feature_num,3])
    color[:] = [200, 0, 200]

    im1_g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_g = cv2.bilateralFilter(im1_g, 3, 50, 50)
    im2_g = cv2.bilateralFilter(im2_g, 3, 50, 50)

    p0 = cv2.goodFeaturesToTrack(im1_g, mask=None, **feature_params)
    mask = np.zeros_like(im1)
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1_g, im2_g, p0, None,  **lk_params)

    good_new = p1[st == 1] 
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)): 
        a, b = new.ravel() 
        c, d = old.ravel() 
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)       
        right = cv2.circle(im2, (a, b), 5,  color[i].tolist(), -1) 
          
    img = cv2.add(im2, mask)
    return img


def build_model():
    print("building model...")
    model = Sequential()
    model.add(Conv2D(filteres = 32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(1024,512,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filteres = 64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(1024,512,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model


def main():
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    op_flow = flow(im1, im2, 100)
    build_model()


if __name__ == '__main__':
    main()