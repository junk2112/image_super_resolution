from common import *
from scipy import spatial
import sys
sys.setrecursionlimit(10000)

class SuperResolutor:
    def __init__(self, src_path, LR_set_step, patch_size):
        self.downscale_odds = 2
        self.src_path = src_path
        self.LR_set_step = LR_set_step
        self.patch_size = patch_size
        try:
            self.src_image = cv2.imread(src_path)
        except:
            raise Exception("Can't read image from '%s'" % (src_path))

    def _gen_LR_set(self, image, LR_set_max_scale, LR_set_step):
        if image is None:
            raise Exception("Image is None")
        result = {}
        if 1 + LR_set_step > LR_set_max_scale:
            raise Exception("1 + LR_set_step > LR_set_max_scale")
        for scale in range(1 + LR_set_step, LR_set_max_scale + LR_set_step, LR_set_step):
            result[scale] = downscale(image, scale)
        return result

    def _get_sift_descriptor(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, d = sift.detectAndCompute(image, None)
        print(d.shape)

    def _get_surf_descriptor(self, image):
        surf = cv2.xfeatures2d.SURF_create()
        kp, d = surf.detectAndCompute(image, None)
        print(d.shape)

    def _crop(self, image, original, scale, patch_size):
        result = []
        height, width, channels = image.shape
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                cropped = get_subsample(image, (i, i+patch_size), (j, j+patch_size))
                result.append({
                    "scale": scale,
                    "cropped": cropped,
                    "descriptor": image_to_tuple(cropped),
                    "original": get_subsample(original, (i, i+patch_size), (j, j+patch_size), scale),
                    "coords": (i, j)
                    })
        # self._get_sift_descriptor(cv2.cvtColor(result[0]["cropped"], cv2.COLOR_BGR2GRAY))
        return result

    def _replace_parts(self, patches, result_scale):
        # result = np.zeros(self.src_image, dtype=np.bool)
        result = gray(upscale(self.src_image, result_scale))
        for patch in patches:
            current_scale = patch["replace_to"]["scale"]
            replacement = patch["replace_to"]["original"]
            x, y = patch["coords"]
            if current_scale != self.scale:
                replacement = upscale(replacement, result_scale/current_scale)
            # elif current_scale > self.scale:
            #     replacement = downscale(replacement, current_scale/self.scale)
            result = replace(result, gray(replacement), x*result_scale, y*result_scale)
        return result

    def scale(self, result_scale):
        LR_set = self._gen_LR_set(self.src_image, result_scale*self.downscale_odds, self.LR_set_step)
        LR_patches = []
        for scale, image in LR_set.items():
            LR_patches += self._crop(image, self.src_image, scale, self.patch_size)
        orig_patches = self._crop(self.src_image, self.src_image, 1, self.patch_size)
        tree = spatial.KDTree([item["descriptor"] for item in LR_patches])
        for i, patch in enumerate(orig_patches):
            distance, index = tree.query([patch["descriptor"]])
            orig_patches[i]["replace_to"] = LR_patches[index]
        result = self._replace_parts(orig_patches, result_scale)
        show([self.src_image, result])
        return result



path = "../datasets/Set5/image_SRF_2/img_002_SRF_2_HR.png"
SuperResolutor(path, LR_set_step=1, patch_size=4).scale(2)
