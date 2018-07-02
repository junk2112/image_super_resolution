import numpy as np
import cv2
import os
from threading import Thread, Timer
import scipy.stats as st
from functools import reduce

import time


def show(images):
    if isinstance(images, list):
        for i, image in enumerate(images):
            cv2.imshow("image_%d" % (i), image)
    else:
        cv2.imshow("image", images)
    cv2.waitKey(0)


def downscale(image, scale):
    return cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)


def upscale(image, scale, nearest=False):
    inter = cv2.INTER_CUBIC if not nearest else cv2.INTER_NEAREST
    result = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=inter)
    return result


def get_subsample(image, X, Y, scale=1):
    dst_size_x = round((X[1] - X[0]) * scale)
    dst_size_y = round((Y[1] - Y[0]) * scale)
    dst_x0, dst_x1 = round(X[0] * scale), round(X[1] * scale)
    dst_y0, dst_y1 = round(Y[0] * scale), round(Y[1] * scale)
    if dst_x1 - dst_x0 > dst_size_x:
        dst_x1 -= 1
    if dst_y1 - dst_y0 > dst_size_y:
        dst_y1 -= 1
    if dst_x1 - dst_x0 < dst_size_x:
        dst_x1 += 1
    if dst_y1 - dst_y0 < dst_size_y:
        dst_y1 += 1
    shape = image[dst_x0:dst_x1, dst_y0:dst_y1].shape
    if shape[0] < shape[1]:
        dst_y1 -= abs(shape[1] - shape[0])
    elif shape[0] > shape[1]:
        dst_x1 -= abs(shape[1] - shape[0])
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
    return tuple([item / 255. for item in result])


def replace(image, replace_data, x, y):
    step_x, step_y = replace_data.shape
    image[x:x + step_x, y:y + step_y] = replace_data
    return image


def gauss(img, ksize, s):
    return cv2.GaussianBlur(img, None, ksize, s, s)


def median(img, ksize):
    return cv2.medianBlur(img, ksize)

def get_concat(*args):
    (h0, h1, w0, w1), source, bicubic, result, hr = args[:5]
    concat = np.concatenate(args[2:], 1)
    concat[:1, :, :] = 255
    concat = concat[:1,:,:]

    windows = []
    w_scale = 3
    for item in list(args[2:]):
        h, w, _ = item.shape
        win = item[int(h * h0):int(h * h1), int(w * w0):int(w * w1), :]
        windows.append(upscale(win, w_scale, True))

    # win_width = windows[0].shape[1] * len(windows) + 100
    # concat = concat[:, :win_width, :]

    cut_width = int(windows[0].shape[1] / len(windows))
    cut = windows[0][:, 0:cut_width, :]
    for i, win in enumerate(windows[1:]):
        cut = np.concatenate((cut, win[:, (i + 1) * cut_width: (i + 2) * cut_width, :]), 1)
        cut[:, (i + 1) * cut_width - 1: (i + 1) * cut_width, :] = np.array([0, 0, 255])
    cut_back = np.array([[[255, 255, 255] for _ in range(concat.shape[1])] for _ in range(windows[0].shape[0])])
    cut_back[:, int(concat.shape[1] / 2 - cut.shape[1] / 2): int(concat.shape[1] / 2 + cut.shape[1] / 2), :] = cut

    lower = np.array([[[255, 255, 255] for _ in range(concat.shape[1])] for _ in range(windows[0].shape[0])])
    for i, win in enumerate(windows):
        h, w, _ = result.shape
        win[:, 0:1, :] = np.array([0, 0, 255])
        win[:, -1:, :] = np.array([0, 0, 255])
        win[0:1, :, :] = np.array([0, 0, 255])
        win[-1:, :, :] = np.array([0, 0, 255])
        lower[:, i * w + int((-win.shape[1] + w) / 2): int((win.shape[1] + w) / 2) + i * w, :] = win
        concat[int(h * h0):int(h * h0) + 1, int(w * w0) + i * w:int(w * w1) + i * w, :] = np.array([0, 0, 255])
        concat[int(h * h1) - 1:int(h * h1), int(w * w0) + i * w:int(w * w1) + i * w, :] = np.array([0, 0, 255])
        concat[int(h * h0):int(h * h1), int(w * w0) + i * w:int(w * w0) + i * w + 1, :] = np.array([0, 0, 255])
        concat[int(h * h0):int(h * h1), int(w * w1) - 1 + i * w:int(w * w1) + i * w, :] = np.array([0, 0, 255])
    concat = np.concatenate((concat, lower), 0)

    top = np.array([[[255, 255, 255] for _ in range(concat.shape[1])] for _ in range(source.shape[0])])
    top[:, int(top.shape[1] / 2 - source.shape[1] / 2):int(top.shape[1] / 2 + source.shape[1] / 2), :] = source

    concat = np.concatenate((top, concat), 0)
    concat = np.concatenate((concat, cut_back), 0)

    return concat

def list_dir(path):
    items = os.listdir(path)
    items = [os.path.join(path, item) for item in items]
    return items

def list_files(path, _filter=lambda item: True):
    return [item for item in list_dir(path) if os.path.isfile(item) if _filter(item)]

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result, (te - ts) * 1000
    return timed