import sys
import copy
import os

from pyflann import *

from Metrics import PSNR
from common import *


sys.setrecursionlimit(100000)


class SuperResolutor:

    def __init__(self, src, LR_set_step, downscale_multiplier, patch_size, patch_step, kernel_sigma):
        self.downscale_multiplier = downscale_multiplier
        self.LR_set_step = LR_set_step
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.kernel_sigma = kernel_sigma
        self.kernel = None
        try:
            if isinstance(src, str):
                self.src_image = cv2.imread(src)
                print(type(self.src_image))
            elif str(type(src)) == "<class 'numpy.ndarray'>" or \
                    str(type(src)) == "<type 'numpy.ndarray'>":
                self.src_image = src
            else:
                raise Exception
        except:
            raise Exception("Can't read image from '%s'" % (src))
        self.orig_patches = self._crop(
            self.src_image, self.src_image, 1)

    def _gen_LR_set(self, image, LR_set_max_scale, LR_set_step):
        if image is None:
            raise Exception("Image is None")
        result = {}
        if 1 + LR_set_step > LR_set_max_scale:
            raise Exception("1 + LR_set_step > LR_set_max_scale")
        scale = 1 + LR_set_step
        while scale <= LR_set_max_scale:
            result[scale] = downscale(image, scale)
            LR_set_step *= 2
            scale += LR_set_step
            scale = round(scale, 3)
        # print(sorted(result.keys()))
        return result

    def _sum_part(self, image, intensity, replace_data, x, y):
        step_x, step_y, _ = replace_data.shape
        image[x:x + step_x, y:y + step_y] += replace_data.astype("float32")
        intensity[x:x + step_x, y:y + step_y] += self.kernel

    def _crop(self, image, original, scale):
        result = []
        height, width, channels = image.shape
        for i in range(0, height - self.patch_size + self.patch_step, self.patch_step):
            for j in range(0, width - self.patch_size + self.patch_step, self.patch_step):
                cropped = get_subsample(
                    image, (i, i + self.patch_size), (j, j + self.patch_size))
                result.append({
                    "scale": scale,
                    "cropped": cropped,
                    "descriptor": image_to_tuple(cropped),
                    "original": get_subsample(
                        original, 
                        (i, i + self.patch_size), 
                        (j, j + self.patch_size), 
                        scale,
                    ),
                    "coords": (i, j)
                })
        return result

    def _replace_parts(self, patches, result_scale):
        result = upscale(self.src_image, result_scale).astype("float32")
        result[:, :, :] = 0
        intensity = copy.deepcopy(result)
        for patch in patches:
            current_scale = patch["replace_to"]["scale"]
            replacement = patch["replace_to"]["original"]
            if replacement.shape[0] != replacement.shape[1]:
                # print("skip1")
                continue

            upscale_coefficient = float(self.kernel.shape[0]) / replacement.shape[0] \
                if self.kernel is not None \
                else float(result_scale) / current_scale
            if current_scale != result_scale:
                replacement = upscale(
                    replacement, 
                    upscale_coefficient,
                )
            if self.kernel is None:
                s = replacement.shape[0]
                self.kernel = gkern(s, self.kernel_sigma)
                self.kernel = np.transpose(np.array([self.kernel,
                                                     self.kernel,
                                                     self.kernel]))
            if replacement.shape != self.kernel.shape:
                # print("skip2")
                continue
            replacement = replacement * self.kernel
            x, y = patch["coords"]
            self._sum_part(result, intensity, replacement,
                           int(x * result_scale), int(y * result_scale))
        result =  np.round(result / intensity)
        return result.astype("uint8"), intensity

    def scale(self, result_scale, is_show=False):
        LR_set = self._gen_LR_set(
            self.src_image, result_scale * self.downscale_multiplier, self.LR_set_step)
        LR_patches = []
        for scale, image in LR_set.items():
            LR_patches += self._crop(image, self.src_image, scale)

        channels = self.src_image.shape[2] if len(
            self.src_image.shape) == 3 else 1
        get_valid_patches = lambda patches: [item for item in patches if len(
            item["descriptor"]) == self.patch_size**2 * channels]

        # get_invalid_patches = lambda patches: [item for item in patches if len(
        #     item["descriptor"]) != (self.patch_size**2) * channels]
        # print([item["coords"] for item in get_invalid_patches(self.orig_patches)])
        # print(self.src_image.shape)

        LR_patches = get_valid_patches(LR_patches)
        self.orig_patches = get_valid_patches(self.orig_patches)

        dataset = np.asarray([np.asarray(item["descriptor"])
                              for item in LR_patches])
        queryset = np.asarray([np.asarray(item["descriptor"])
                               for item in self.orig_patches])

        replace_to, dist = FLANN().nn(dataset, queryset, 1, algorithm="kmeans")
        replace_to = [LR_patches[i] for i in replace_to]

        for i, patch in enumerate(self.orig_patches):
            self.orig_patches[i]["replace_to"] = replace_to[i]
        result, intensity_map = self._replace_parts(
            self.orig_patches, result_scale)

        upscaled = upscale(self.src_image, result_scale)

        # TODO: FIX THIS SHIT
        # print(PSNR.evaluate(result, upscaled))
        result[-2:-1, 0:-1] = upscaled[-2:-1, 0:-1]
        result[0:-1, -2:-1] = upscaled[0:-1, -2:-1]
        print(PSNR.evaluate(result, upscaled))

        if is_show:
            # print(np.mean(upscaled), np.mean(result))
            intensity_map = (255 * intensity_map / np.max(intensity_map)).astype("int8")
            # intensity_map[intensity_map == 0.0] = 255
            # intensity_map[intensity_map != 255] = 0
            # s1 = cv2.Sobel(result,cv2.CV_8U,1,1,ksize=5)
            # s2 = cv2.Sobel(upscaled,cv2.CV_8U,1,1,ksize=5)
            show([
                # np.concatenate((s1, s2, np.abs(s1 - s2)), 1),
                intensity_map, np.concatenate((upscaled, result), 1)])
        return result


if __name__ == '__main__':
    # path = "../samples/img_002_SRF_2_LR.png"
    # path = "../samples/img_001_SRF_2_LR.png"
    # path = "../samples/img_013_SRF_2_LR.png"
    # path = "../samples/img_006_SRF_2_LR.png"
    path = "../datasets/Urban100_SR/image_SRF_2/img_002_SRF_2_LR.png"

    source = cv2.imread(path)
    scale = 2
    result = SuperResolutor(source,
                   LR_set_step=0.7,
                   downscale_multiplier=50,
                   patch_size=3,
                   patch_step=1,
                   kernel_sigma=0.7).scale(scale, False)

    result_debug_path = '../results/debug/'
    cv2.imwrite(os.path.join(result_debug_path, 'result.png'), result)
    cv2.imwrite(os.path.join(result_debug_path, 'upscaled.png'), upscale(source, scale))
    # SuperResolutor(cv2.imread(path),
    #                LR_set_step=0.375,
    #                downscale_multiplier=20,
    #                patch_size=5,
    #                patch_step=1,
    #                kernel_sigma=1).scale(2, True)
