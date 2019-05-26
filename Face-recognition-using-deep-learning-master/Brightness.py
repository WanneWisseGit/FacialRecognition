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
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    im = Image.fromarray(frame)
    a = calculate_brightness(im)
    print(a)
    c = -0.3/math.log10(a)
    print(c)
    g = np.array(im)
    gamma_corrected = exposure.adjust_gamma(g, c)

    gamma_corrected_image = Image.fromarray(gamma_corrected)
    # gamma_corrected_image.show()

    i = exposure.equalize_adapthist(gamma_corrected)
    #ii = Image.fromarray((i * 255).astype(np.uint8))

    # x = Image.fromarray(g)
    # j = exposure.equalize_adapthist(g)
    # jj = Image.fromarray((j * 255).astype(np.uint8))



    # b = Image.fromarray(frame)
    # a = calculate_brightness(b)
    # print(a)
    # c = -0.3/math.log10(a)
    # print(c)
    # frame1 = exposure.adjust_gamma(frame, c)
    # frame2 = exposure.equalize_hist(frame1)
    # #print(gamma_corrected)
    # print("asddasasdsadasdasdds") 
    # #print(gamma_corrected)
    # cv2.imwrite("test/bright.jpg",  frame1)

    # Display the resulting frame
    cv2.imshow('frame',i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# img=cv2.imread('images\jape.jpg')
# b = Image.fromarray(img)
# a = calculate_brightness(b)
# print(a)
# c = -0.3/math.log10(a)
# print(c)
# gamma_corrected = exposure.adjust_gamma(img, c)
# img_eq = exposure.equalize_hist(gamma_corrected)

# cv2.imshow('orginal', img)
# cv2.imshow('gamma and histogram', img_eq)
# cv2.imshow('gamma', gamma_corrected)


cv2.waitKey()
cv2.destroyAllWindows()