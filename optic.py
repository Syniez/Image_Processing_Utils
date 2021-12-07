import numpy as np
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import MaxPool2D


def flow(im1, im2, step):
    h, w = im1.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(im1, im2, 0, 0.5, 3, 15, 3, 5, 1.1, 0)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    res = np.zeros((im1.shape[0], im1.shape[1], 3), np.uint8)
   
    cv2.polylines(res, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(res, (x1, y1), 1, (0, 255, 0), -1)
    #res  =cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res


def build_model():
    print("building model...")
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(1024,512,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(1024,512,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(12))

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


def train(model, x, y):
    model.fit(x, y, epochs=50, verbose=1)


def test(model, x):
    y_pred = model.predict(x, verbose=1)
    y_pred = y_pred.reshape(3,4)
    print(y_pred)


def main():
    im1_path = "./Blender/data/a/cam00001.jpg"
    im2_path = "./Blender/data/a/cam10001.jpg"
    
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    op_flow = flow(im1, im2, 16)
    op_flow = op_flow / 255

    op_flow = op_flow.reshape((1, 1024, 512, 3))

    model = build_model()

    t = np.array([[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]])
    t = t.reshape((1,1*12))
    train(model, op_flow, t)


    test1 = cv2.imread("./Blender/data/class_re/cam00002.jpg")
    test2 = cv2.imread("./Blender/data/class_re/cam10030.jpg")
    op_flow2 = flow(test1, test2, 16)
    op_flow2 = op_flow2 / 255
    op_flow2 = op_flow2.reshape((1, 1024, 512, 3))

    test(model, op_flow2)


if __name__ == '__main__':
    main()
