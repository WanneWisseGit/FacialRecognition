import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img1=cv.imread('images\Jari.jpg')
img2=cv.cvtColor(img1,cv.COLOR_RGB2HSV)
img2[:,:,2] = cv.equalizeHist(img2[:,:,2])
img4=cv.cvtColor(img2,cv.COLOR_HSV2RGB)



im1=cv.imread('images\Jari.jpg')
im_input_Gamma=cv.cvtColor(im1,cv.COLOR_RGB2HSV)

cv.imshow('orginal',im1[:,:,:])

#r,c,d=im2.shape
gamma1 = 0.5
lookUpTable1 = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable1[0,i] = np.clip(pow(i / 255.0, gamma1) * 255.0, 0, 255)

im_input_Gamma[:,:,2] = cv.LUT(im_input_Gamma[:,:,2], lookUpTable1)
between=cv.cvtColor(im_input_Gamma,cv.COLOR_HSV2RGB)
im_input_HIS=between.copy()
im_input_HIS = cv.cvtColor(im_input_HIS,cv.COLOR_RGB2HSV)

im_input_HIS[:,:,2] = cv.equalizeHist(im_input_HIS[:,:,2])
Ouput=cv.cvtColor(im_input_HIS,cv.COLOR_HSV2RGB)
cv.imshow('Gamma Corrected Image',img4)
cv.imshow('Histogram Equalized Image',between)
cv.imshow('Histogram Equalized Image',im_input_HIS)
cv.imshow('Histogram',Ouput)
#cv.moveWindow('messi',100,100)


cv.waitKey()
cv.destroyAllWindows()