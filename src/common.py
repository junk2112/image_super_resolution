import numpy as np
import cv2

def show(images):
    if isinstance(images, list):
        for i, image in enumerate(images):
            cv2.imshow("image_%d"%(i), image)
    else:
        cv2.imshow("image", images)
    cv2.waitKey(0)

def downscale(image, scale):
    return cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)

def upscale(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def get_subsample(image, X, Y, scale=1):
    return image[X[0]*scale:X[1]*scale, Y[0]*scale:Y[1]*scale]

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_to_tuple(image):
    result = []
    for row in gray(image):
        result += list(row)
    return tuple(result)

def replace(image, replace_data, x, y):
    step_x, step_y = replace_data.shape
    image[x:x+step_x, y:y+step_y] = replace_data
    return image