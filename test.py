import matplotlib.pyplot as plt
import numpy as np
import cv2

# rgb 별 모든 픽셀의 평균값의 차이
# intensity의 평균값

im_l = cv2.imread('/home/leejongsung/vcl/Jong/ossumpia_data/data/09_07/1/0000.png')
im_r = cv2.imread('/home/leejongsung/vcl/Jong/ossumpia_data/data/09_07/2/0000.png')

print("왼쪽 영상 rgb 평균값 :", np.mean(im_l[:, :, 0]), np.mean(im_l[:, :, 1]), np.mean(im_l[:, :, 2]))
print("오른쪽 영상 rgb 평균값 :", np.mean(im_r[:, :, 0]), np.mean(im_r[:, :, 1]), np.mean(im_r[:, :, 2]))

dif = abs(im_l - im_r)
print("차 영상 rgb 평균값 :", np.mean(dif[:, :, 0]), np.mean(dif[:, :, 1]), np.mean(dif[:, :, 2]))