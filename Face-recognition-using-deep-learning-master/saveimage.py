import sys
import math
import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure
from PIL import ImageOps, Image
def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

im = Image.open("images/wanne.jpg")

#b = Image.fromarray(im)
a = calculate_brightness(im)
print(a)
c = -0.3/math.log10(a)
print(c)
g = np.array(im)
gamma_corrected = exposure.adjust_gamma(g, c)

gamma_corrected_image = Image.fromarray(gamma_corrected)
gamma_corrected_image.show()

i = exposure.equalize_adapthist(gamma_corrected)
ii = Image.fromarray((i * 255).astype(np.uint8))
ii.save("pictures/hello1.png")

x = Image.fromarray(g)
j = exposure.equalize_adapthist(g)
jj = Image.fromarray((j * 255).astype(np.uint8))
jj.save("pictures/hello2.png")

# gamma_corrected_image.save("pictures/hello.png")
# hist = ImageOps.equalize(gamma_corrected_image)
# im2 = Image.open("pictures/hello.png")
# hist2 = ImageOps.equalize(im2)
# histogram_corrected = exposure.equalize_adapthist(gamma_corrected)

# histogram_corrected_image = Image.fromarray(histogram_corrected, mode='RGB')
ii.show()
gamma_corrected_image.show()


# cv2.imshow('orginal', im)
# cv2.imshow('gamma and histogram', img_eq)
# cv2.imshow('gamma', gamma_corrected)
# a.save("test/bright.jpg")
# # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# # # equalize the histogram of the Y channel
# # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# # # convert the YUV image back to RGB format
# # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# # cv2.imwrite("test/bright.jpg",  img_output)