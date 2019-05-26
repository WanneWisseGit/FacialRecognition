import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure


img=cv2.imread('images/wanne.png')

gamma_corrected = exposure.adjust_gamma(img, 0.5)
logarithmic_corrected = exposure.adjust_log(gamma_corrected, 1)
img_eq = exposure.equalize_hist(img)


cv2.imshow('ae', img_eq)
cv2.imshow('aes', logarithmic_corrected)


cv2.waitKey()
cv2.destroyAllWindows()