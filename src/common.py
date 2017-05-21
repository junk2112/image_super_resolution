import numpy as np
import cv2
from threading import Thread, Timer
import scipy.stats as st

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
    result = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # print(image.shape, result.shape, scale)
    return result

def get_subsample(image, X, Y, scale=1):
    # print(X[0]*scale, X[1]*scale, Y[0]*scale, Y[1]*scale)
    return image[int(X[0]*scale):int(X[1]*scale), int(Y[0]*scale):int(Y[1]*scale)]

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

kernels = {}
def image_to_tuple(image):
    result = []
    # print(image.shape)
    global kernels
    if image.shape[0] not in kernels:
        kernels[image.shape[0]] = gkern(image.shape[0])
    for i, row in enumerate(gray(image)):
        result += [item/255.0 for item, g in zip(list(row), kernels[image.shape[0]][i])]
    # [means, stds] = cv2.meanStdDev(image)
    # result += [item for item in list(means)]
    # result += [item for item in list(stds)]
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
