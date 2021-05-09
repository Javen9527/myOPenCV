# import cv
PYTHONPATH = 'D:\Git\myrepo\opencv\opencv-precompiled\opencv\build\python'
import cv2
print('************ opencv version: ' + cv2.__version__ +' ************')

import numpy as np

from mylib import stackImg as stackImg

# read pic
"""
img = cv2.imread('resources/Javen.jpg')
"""
img = cv2.imread('resources/wanzi.PNG')

# read video
'''
cap = cv2.VideoCapture('resources\Viviana.mp4')
'''
# from computer camera
"""
cap = cv2.VideoCapture(0)
cap.set(3, 400) # width
cap.set(4, 600) # height
cap.set(10, 200) # light

while True:
    success, img = cap.read()
    imgCanny = cv2.Canny(img, 100, 100)
    cv2.imshow('window', imgCanny)
    if(cv2.waitKey(50) & 0xFF == ord('q')):
        break
"""

# image handle function
"""
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGuss = cv2.GaussianBlur(img, (7,7), 0)
imgCanny = cv2.Canny(img, 100, 100)

kernel = np.ones((5,5), np.uint8)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iteratons=1)
"""

# resize img
"""
print(img.shape) # height width 图是先行后列
imgResize = cv2.resize(img, (200, 300)) # width height 点是坐标点xy（即先列后行）
imgCropped = img[0:200, 200:500]
"""

# shape and text
"""
img = np.zeros((512,512,3), np.uint8)
img[:] = 1, 0, 0

dot = ((int)(img.shape[1]/2), (int)(img.shape[0]/2))
cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 3)
cv2.rectangle(img, (0,0), dot, (0,255,0))
cv2.circle(img, dot, 50, (255,0,0))

cv2.putText(img, "Javen", dot, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
"""

# 平面倾斜校正
"""
pts1 = np.float32([], [], [], [])
pts2 = np.float32([0,0], [width,0], [0,height], [width,height])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

imgPerspec = cv2.wrapPerspective(img, matrix, (widht, height))
"""

# 图像多窗口显示
"""
imgHor = np.hstack((img, img))
imgVer = np.vstack((img, img))

imgStack = stackImg.stackImages(0.5, ([img,img,img], [img,img,img]))
"""
cv2.imshow('winname', img)
cv2.waitKey(0)








