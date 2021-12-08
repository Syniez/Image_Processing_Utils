from math import sin, cos
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import MaxPool2D


def rotMat(theta):
    Rx = np.array([
        [1, 0, 0],
        [0, cos(theta[0]), -sin(theta[0])],
        [0, sin(theta[0]), cos(theta[0])]
    ])
    Ry = np.array([
        [cos(theta[1]), 0, sin(theta[1])],
        [0, 1, 0],
        [-sin(theta[1]), 0, cos(theta[1])]
    ])
    Rz = np.array([
        [cos(theta[2]), -sin(theta[2]), 0],
        [sin(theta[2]), cos(theta[2]), 0],
        [0, 0, 1]
    ])
    
    return Rx @ Ry @ Rz


def flow(im1, im2, step):
    h, w = im1.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    flow = cv2.calcOpticalFlowFarneback(im1, im2, 0, 0.5, 3, 15, 3, 5, 1.1, 0)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    res = np.zeros((im1.shape[0], im1.shape[1], 3), np.uint8)
   
    cv2.polylines(res, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(res, (x1, y1), 1, (0, 255, 0), -1)

    return res


def data_load(path1, path2):
    print("load data...")

    im1s = sorted(os.listdir(path1))
    im2s = sorted(os.listdir(path2))

    flows = np.empty((len(im1s)-1, 512, 1024, 3))
    for i in range(len(im1s)-1):
        im1 = cv2.imread(path1 + im1s[i], 2)
        im2 = cv2.imread(path2 + im2s[i], 2)

        op_flow = flow(im1, im2, 16)
        op_flow = (op_flow / 255)
        flows[i] = op_flow
    
    y = []
    cam1_r = [90, 0, 180]
    cam2_r = [90, -10, 180]
    for i in range(30):
        cam1_r[2] += 1
        cam2_r[0] -= 0.5
        cam2_r[2] -= 0.5

        y.append([cam1_r[0] - cam2_r[0], cam1_r[1] - cam2_r[1], cam1_r[2] - cam2_r[2]])
    y = np.array(y)
    
    return flows, y



def build_model(x):
    print("building model...")
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(1024,512,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3))

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


def train(model, x, y):
    model.fit(x, y, epochs=20, verbose=1)


def test(model, x):
    y_pred = model.predict(x, verbose=1)
    #y_pred = y_pred.reshape(3,4)
    print(y_pred)


def main():
    im1_path = "./Blender/data/test/cam0/"
    im2_path = "./Blender/data/test/cam1/"

    x, y = data_load(im1_path, im2_path)


    model = build_model(x)
    print(y[29])
    train(model, x, y)

    #test(model, op_flow2)
    test1 = cv2.imread("./Blender/data/test/cam0/cam00031.jpg")
    test2 = cv2.imread("./Blender/data/test/cam1/cam10031.jpg")
    test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
    test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)
    op_flow2 = flow(test1, test2, 16)
    op_flow2 = op_flow2 / 255
    op_flow2 = op_flow2.reshape((1, 1024, 512, 3))

    test(model, op_flow2)


if __name__ == '__main__':
    main()