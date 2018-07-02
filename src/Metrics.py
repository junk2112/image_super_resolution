import numpy as np
import math
from scipy.ndimage.filters import correlate
from skimage.measure import compare_ssim as ssim


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

    @staticmethod
    def evaluate(image, source):
        raise NotImplementedError()


class PSNR(Metric):

    @staticmethod
    @check_sizes
    def evaluate(image, source):
        channels = image.shape[2] if len(image.shape) == 3 else 1
        mse = float(np.sum((image - source) ** 2))
        if mse == 0:
            # raise Exception("Same images")
            return -1
        mse /= image.shape[0] * image.shape[1] * channels
        # 65025 is 255**2
        return 10 * math.log((65025) / mse, 10)


class SSIM(Metric):

    @staticmethod
    @check_sizes
    def evaluate(img1, img2):
        return ssim(img1, img2, multichannel=True)


class VIF(Metric):

    @staticmethod
    def __get_gaussian_kernel(N=15, sigma=1.5):
        (H, W) = ((N - 1) / 2, (N - 1) / 2)
        std = sigma
        (y, x) = np.mgrid[-H:H + 1, -W:W + 1]
        arg = -(x * x + y * y) / (2.0 * std * std)
        h = np.exp(arg)
        index = h < np.finfo(float).eps * h.max(0)
        h[index] = 0
        sumh = h.sum()
        if sumh != 0:
            h = h / sumh
        return h

    @staticmethod
    def __filter2(B, X, shape='nearest'):
        B2 = np.rot90(np.rot90(B))
        if len(X.shape) < 3:
            return correlate(X, B2, mode=shape)
        else:
            channels = X.shape[2]
            f = [correlate(X[:, :, c], B2, mode=shape) for c in range(channels)]
            return np.array(f)

    def __get_sigma(win, ref, dist, mu1_sq, mu2_sq, mu1_mu2):
        sigma1_sq = VIF.__filter2(win, ref * ref) - mu1_sq
        sigma2_sq = VIF.__filter2(win, dist * dist) - mu2_sq
        sigma12 = VIF.__filter2(win, ref * dist) - mu1_mu2
        (sigma1_sq[sigma1_sq < 0], sigma2_sq[sigma2_sq < 0]) = (0.0, 0.0)
        return (sigma2_sq, sigma12, sigma1_sq)

    @staticmethod
    def __get_normalized(s1s1, s2s2, s1s2):
        g = s1s2 / (s1s1 + 1e-10)
        sv_sq = s2s2 - g * s1s2
        g[s1s1 < 1e-10] = 0
        sv_sq[s1s1 < 1e-10] = s2s2[s1s1 < 1e-10]
        s1s1[s1s1 < 1e-10] = 0
        g[s2s2 < 1e-10] = 0
        sv_sq[s2s2 < 1e-10] = 0
        sv_sq[g < 0] = s2s2[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10
        return (g, sv_sq)

    @staticmethod
    def __get_num(s1s1, sv_sq, sigma_nsq, g):
        normg = (g ** 2) * s1s1 / (sv_sq + sigma_nsq)
        snr = np.log10(1.0 + normg).sum()
        return snr

    @staticmethod
    def __get_den(s1s1, sigma_nsq):
        snr = np.log10(1.0 + s1s1 / sigma_nsq)
        return snr.sum()

    @staticmethod
    def __get_num_den_level(ref, dist, scale):
        sig = 2.0
        N = (2.0 ** (4 - scale + 1.0)) + 1.0
        win = VIF.__get_gaussian_kernel(N, N / 5.0)
        if scale > 1:
            ref = VIF.__filter2(win, ref)
            dist = VIF.__filter2(win, dist)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
        (mu1, mu2) = (VIF.__filter2(win, ref), VIF.__filter2(win, dist))
        (m1m1, m2m2, m1m2) = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
        (s2s2, s1s2, s1s1) = VIF.__get_sigma(win, ref, dist, m1m1, m2m2, m1m2)
        (g, svsv) = VIF.__get_normalized(s1s1, s2s2, s1s2)
        (num, den) = (VIF.__get_num(s1s1, svsv, sig, g), VIF.__get_den(s1s1, sig))
        return (num, den)


    @staticmethod
    @check_sizes
    def evaluate(reference, query):
        (ref, dist) = (reference.astype('double'), query.astype('double'))
        zipped = map(lambda x: VIF.__get_num_den_level(ref, dist, x), range(1, 5))
        (nums, dens) = zip(*zipped)
        value = sum(nums) / sum(dens)
        return value


import numpy
import scipy.signal
import scipy.ndimage


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp


class MetricAggregator:

    metrics = [PSNR, SSIM, VIF]

    @staticmethod
    def evaluate(img1, img2):
        result = {}
        for m in MetricAggregator.metrics:
            result[m.__name__] = float('%.3f' % m.evaluate(img1, img2))
        return result


if __name__ == "__main__":
    import cv2
    img1 = cv2.imread("/Users/andrew/git/image_super_resolution/results/debug/result.png")
    img2 = cv2.imread("/Users/andrew/git/image_super_resolution/results/debug/hr_image.png")
    print(VIF().evaluate(img1, img2))
    print(vifp_mscale(img2, img1))
