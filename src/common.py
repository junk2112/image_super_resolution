import numpy as np
import cv2
from threading import Thread, Timer
import scipy.stats as st
from functools import reduce


def show(images):
    if isinstance(images, list):
        for i, image in enumerate(images):
            cv2.imshow("image_%d" % (i), image)
    else:
        cv2.imshow("image", images)
    cv2.waitKey(0)


def downscale(image, scale):
    return cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)


def upscale(image, scale):
    result = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
    # print(image.shape, result.shape, scale)
    return result


def get_subsample(image, X, Y, scale=1):
    dst_size_x = int((X[1] - X[0]) * scale)
    dst_size_y = int((Y[1] - Y[0]) * scale)
    dst_x0, dst_x1 = int(X[0] * scale), int(X[1] * scale)
    dst_y0, dst_y1 = int(Y[0] * scale), int(Y[1] * scale)
    if dst_x1 - dst_x0 > dst_size_x:
        dst_x1 -= 1
    if dst_y1 - dst_y0 > dst_size_y:
        dst_y1 -= 1
    return image[dst_x0:dst_x1, dst_y0:dst_y1]


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gkern(kernlen, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel /= np.max(kernel)
    return kernel


def image_to_tuple(image):
    result = []
    for color in np.transpose(image):
        result += reduce(lambda acc, x: list(acc) + list(x), color)
    return tuple(result)


def replace(image, replace_data, x, y):
    step_x, step_y = replace_data.shape
    image[x:x + step_x, y:y + step_y] = replace_data
    return image


def gauss(img, ksize, s):
    return cv2.GaussianBlur(img, None, ksize, s, s)


def median(img, ksize):
    return cv2.medianBlur(img, ksize)
