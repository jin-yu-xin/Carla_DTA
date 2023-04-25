import glob
import os
import sys
import math
import numpy as np
import time
import cv2

def funOutputImgProperties(img):
    print("properties:shape:{},size:{},dtype:{}".format(img.shape,img.size,img.dtype))

converted = cv2.imread('./output/converted/1-00000.png')
raw = cv2.imread('./output/raw/1-00000.png')

mask = cv2.inRange(converted, (142, 0, 0), (142, 0, 0))
ref_image = cv2.bitwise_and(raw, raw, mask=mask)
# funOutputImgProperties(mask)
# funOutputImgProperties(converted)
# funOutputImgProperties(raw)

# # print(converted)

cv2.imshow('converted', converted)
cv2.imshow('raw', raw)
cv2.imshow('mask', mask)
cv2.imshow('ref', ref_image)
cv2.waitKey(0)


