import sys
import math
import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure
from PIL import Image


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

a = calculate_brightness(image)
b = -0.3/math.log10(a)


gamma_corrected = exposure.adjust_gamma(img, b)
img_eq = exposure.equalize_hist(gamma_corrected)

