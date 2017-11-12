import numpy as np
import math


def check_sizes(f):
    def check(*args, **kwargs):
        try:
            image1_shape, image2_shape = args[0].shape, args[1].shape
        except AttributeError:
            raise Exception("Objects must be numpy arrays")
        if image1_shape != image2_shape:
            raise Exception("Shapes must be equal")
        return f(*args, **kwargs)
    return check


class Metric:

    def evaluate(image, source):
        raise NotImplementedError()


# Peak signal-to-noise ratio
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
class PSNR(Metric):

    @staticmethod
    @check_sizes
    def evaluate(image, source):
        channels = image.shape[2] if len(image.shape) == 3 else 1
        mse = float(np.sum((image - source) ** 2))
        if mse == 0:
            raise Exception("Same images")
        mse /= image.shape[0] * image.shape[1] * channels
        # 65025 is 255**2
        return 10 * math.log((65025) / mse, 10)


if __name__ == "__main__":
    import cv2
    img1 = cv2.imread("../results/result_1/HR.png")
    img2 = cv2.imread("../results/result_1/3-0.500-0.7_30.844.png")
    print(PSNR().evaluate(img1, img2))
