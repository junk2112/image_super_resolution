from common import *
from scipy import spatial
import sys
sys.setrecursionlimit(100000)

class SuperResolutor:
    def __init__(self, src, LR_set_step, downscale_multiplier, patch_size, patch_step):
        self.downscale_multiplier = downscale_multiplier
        self.LR_set_step = LR_set_step
        self.patch_size = patch_size
        self.patch_step_del = patch_size//patch_step
        try:
            if isinstance(src, str):
                self.src_image = cv2.imread(src)
                print(type(self.src_image))
            elif str(type(src)) == "<class 'numpy.ndarray'>":
                self.src_image = src
            else:
                raise Exception
        except:
            raise Exception("Can't read image from '%s'" % (src))
        self.orig_patches = self._crop(self.src_image, self.src_image, 1, self.patch_size)

    def _gen_LR_set(self, image, LR_set_max_scale, LR_set_step):
        if image is None:
            raise Exception("Image is None")
        result = {}
        if 1 + LR_set_step > LR_set_max_scale:
            raise Exception("1 + LR_set_step > LR_set_max_scale")
        scale = 1 + LR_set_step
        while scale <= LR_set_max_scale:
            result[scale] = downscale(image, scale)
            scale += LR_set_step
        print(result.keys())
        return result

    # def _get_sift_descriptor(self, image):
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     kp, d = sift.detectAndCompute(image, None)
    #     print(d.shape)

    # def _get_surf_descriptor(self, image):
    #     surf = cv2.xfeatures2d.SURF_create()
    #     kp, d = surf.detectAndCompute(image, None)
    #     print(d.shape)

    def _crop(self, image, original, scale, patch_size):
        result = []
        height, width, channels = image.shape
        margin = patch_size//self.patch_step_del
        for i in range(0, height-margin, margin):
            for j in range(0, width-margin, margin):
                cropped = get_subsample(image, (i, i+patch_size), (j, j+patch_size))
                result.append({
                    "scale": scale,
                    "cropped": cropped,
                    "descriptor": image_to_tuple(cropped),
                    "original": get_subsample(original, (i, i+patch_size), (j, j+patch_size), scale),
                    "coords": (i, j)
                    })
        return result

    def _replace_parts(self, patches, result_scale):
        result = upscale(self.src_image, result_scale)
        result[:,:,:] = 0
        for patch in patches:
            current_scale = patch["replace_to"]["scale"]
            replacement = patch["replace_to"]["original"]
            x, y = patch["coords"]
            if current_scale != self.scale:
                replacement = upscale(replacement, result_scale/current_scale)
            # result = replace(result, gray(replacement), x*result_scale, y*result_scale)
            result = sum_part(result, replacement, x*result_scale, y*result_scale, self.patch_step_del)
        return result

    def scale(self, result_scale, is_show=False):
        def find_replacements(patches):
            result = []
            for i, patch in enumerate(patches):
                if not i % 1000:
                    print(i, len(patches))
                distance, index = tree.query([patch["descriptor"]])
                result.append(LR_patches[index])
            return result
        LR_set = self._gen_LR_set(self.src_image, result_scale*self.downscale_multiplier, self.LR_set_step)
        LR_patches = []
        for scale, image in LR_set.items():
            LR_patches += self._crop(image, self.src_image, scale, self.patch_size)

        LR_patches = [item for item in LR_patches if len(item["descriptor"]) == self.patch_size**2 + 6]
        self.orig_patches = [item for item in self.orig_patches if len(item["descriptor"]) == self.patch_size**2 + 6]

        tree = spatial.KDTree([item["descriptor"] for item in LR_patches])
        print(len(self.orig_patches), "patches")
        replace_to = find_replacements(self.orig_patches)
        # replace_to = async(4, self.orig_patches, find_replacements, "patches")

        for i, patch in enumerate(self.orig_patches):
            self.orig_patches[i]["replace_to"] = replace_to[i]
        result = self._replace_parts(self.orig_patches, result_scale)
        if is_show:
            show([self.src_image, upscale(self.src_image, result_scale), result])
        return result


if __name__ == '__main__':
    # path = "../datasets/Set14/image_SRF_2/img_013_SRF_2_LR.png"
    # path = "../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png"
    path = "../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png"
    # "patch_size % patch_step == 0" should be True
    SuperResolutor(cv2.imread(path),
        LR_set_step=0.5,
        downscale_multiplier=1.5,
        patch_size=6,
        patch_step=2).scale(2, True)
