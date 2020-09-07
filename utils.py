import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#skew check X
cameraMatrix1 = np.array([[1124.486549, 0.000000, 638.182533], [0.000000, 1124.199635, 372.521819], [0.000000, 0.000000, 1.000000]])
distCoeffs1 = np.array([-0.481509, 0.342094, -0.002188, 0.000046, 0])
cameraMatrix2 = np.array([[1119.833261, 0.000000, 601.986214], [0.000000, 1120.825806, 350.313075], [0.000000, 0.000000, 1.000000]])
distCoeffs2 = np.array([-0.478307, 0.338927, -0.002580, 0.003656, 0])

#imageSize = (1280, 720)
imageSize = (1280, 450)     # because we will use cropped image
R = np.array([[0.999984, 0.000620, -0.005695], [-0.000623, 1.000000, -0.000616], [0.005695, 0.000620, 0.999984]])
T = np.array([-119.380348, -0.104935, 0.124688])
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)

# region of interest
(width,height) = imageSize
#roi1 = np.array([[(498,300), (802,300), (910,height), (390,height)]], dtype=np.int32)
roi1 = np.array([[(498,300), (822,300), (930,height), (390,height)]], dtype=np.int32)
dst = np.array([[(300,0), (1000,0), (1000,height), (300,height)]], dtype=np.float32)
#dst = np.array([[(0,0), (width,0), (width,height), (0,height)]], dtype=np.float32)

# Camera specs
focacl_length = 3.6     # (mm)
base_line = 120         # (mm)


def Calibrate(img):
     map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_32FC1)
     img = cv2.remap(img, map1, map2,  cv2.INTER_NEAREST)
     return img


def Cut_road(img):
    global roi1
    roi1 = roi1.astype('int32')
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,roi1,(255,255,255))
    masked = cv2.bitwise_and(img,mask)
    return masked


def Cut_surround(img):
    global roi1
    roi1 = roi1.astype('int32')
    mask = np.ones_like(img)*255
    cv2.fillPoly(mask,roi1,(0,0,0))
    masked = cv2.bitwise_and(img,mask)
    return masked


def Warning_process(img_pre, img_cur):
    dif = abs(img_cur - img_pre)
    dif_filtered = cv2.inRange(dif, 40, 130)

    pixel_num = dif_filtered.size
    mean = np.mean(dif_filtered)
    threshold = 200
    #thres = (pixel_num*0.03*255)/pixel_num
    thres = (threshold*255*3)/pixel_num

    #print(mean)
    if mean > thres : print("Warning")
    #if mean > thres :   print("Warning !")


def Bird_eye(img, mask):
    global roi1
    roi1 = roi1.astype('float32')
    M = cv2.getPerspectiveTransform(roi1,dst)
    #M = cv2.getPerspectiveTransform(dst,roi1)   #  M_inv
    warp = cv2.warpPerspective(mask,M,(width,height))
    return warp


def Sobel(img, thres):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    img_x = cv2.convertScaleAbs(img_x)
    img_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    img_y = cv2.convertScaleAbs(img_y)
    img = cv2.addWeighted(img_x,1,img_y,1,0)

    ret, thresh = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)
    return thresh


def Canny(img, low_thres, high_thres):
    img = cv2.Canny(img,low_thres,high_thres)
    return img


def Hough(img1, img2, thres):
    lines = cv2.HoughLinesP(img2,1,np.pi/180,thres)  # input img2 should be grayscaleimage.
    line_img = np.zeros_like(img1)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),[255,0,255],2)

    res = cv2.addWeighted(img1,0.4,line_img,0.6,0)
    return res


def SBM(imL, imR):
    #if imL.shape[2] == 3 :  imL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    #if imR.shape[2] == 3 :  imR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    sbm = cv2.StereoBM_create(numDisparities=256, blockSize=15)
    disparity = sbm.compute(imL, imR)
    return disparity


def SGBM_wls(img_L,img_R):
    h,w = img_L.shape
    maxdisp = 192 * 2
    im1 = cv2.rotate(img_L, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im2 = cv2.rotate(img_R, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width = im1.shape[:2]
    imgL = np.zeros((height, width+maxdisp), dtype = "uint8")
    imgR = np.zeros((height, width+maxdisp), dtype = "uint8")

    imgL[0:height, maxdisp:width+maxdisp] = im1
    imgR[0:height, maxdisp:width+maxdisp] = im2

    window_size = 5                         # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely  
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=maxdisp,             # max_disp has to be dividable by 16
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)


    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    #filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.rotate(filteredImg, cv2.ROTATE_90_CLOCKWISE)


    filteredImg = filteredImg[maxdisp:width+maxdisp, 0:height]
    #cv2.imwrite("Not_filtered.png", np.uint8(displ[maxdisp:width+maxdisp, 0:height]))
    return filteredImg


def wls_filtering(depth, im, Lambda, Sigma):
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(Lambda)
    wls_filter.setSigmaColor(Sigma)

    #if im.shape[2] == 3 :   im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    filtered = wls_filter.filter(depth, im)
    return filtered


def feature_flow(im1, im2, feature_num):
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


def grid_flow(im_g, im, flow, black = False):     # im : grayscale image
    step = 20
    h, w = im_g.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5)
    '''vis = cv2.cvtColor(im_g, cv2.COLOR_GRAY2BGR)

    if black:
        vis = np.zeros((h, w, 3), np.uint8)'''
    cv2.polylines(im, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(im, (x1, y1), 1, (200, 0, 200), -1)
    return im