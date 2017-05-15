import numpy as np
import cv2
from threading import Thread, Timer

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
    # print(X[0]*scale, X[1]*scale, Y[0]*scale, Y[1]*scale)
    return image[int(X[0]*scale):int(X[1]*scale), int(Y[0]*scale):int(Y[1]*scale)]

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_to_tuple(image):
    result = []
    for i, row in enumerate(gray(image)):
        result += list(row)
    # print(cv2.meanStdDev(image))
    [means, stds] = cv2.meanStdDev(image)
    result += [item*1.5 for item in list(means)]
    result += [item*1.5 for item in list(stds)]
    return tuple(result)

def replace(image, replace_data, x, y):
    step_x, step_y = replace_data.shape
    image[x:x+step_x, y:y+step_y] = replace_data
    return image

def sum_part(image, replace_data, x, y, patch_step_del):
    step_x, step_y, _ = replace_data.shape
    if replace_data.shape != image[x:x+step_x, y:y+step_y].shape:
        return image
    image[x:x+step_x, y:y+step_y] += replace_data//(patch_step_del**2)
    return image

def gauss(img, ksize, s):
    return cv2.GaussianBlur(img, None, ksize, s, s)

def median(img, ksize):
    return cv2.medianBlur(img, ksize)
