import copy
from pyflann import *

from src.Metrics import MetricAggregator
from src.common import *

sys.setrecursionlimit(100000)


class SuperResolutor:

    def __init__(
            self,
            src,
            LR_set_step,
            downscale_multiplier,
            patch_size, patch_step,
            kernel_sigma,
            dist_threshold_scale,
            exp_scale=True,
    ):
        self.downscale_multiplier = downscale_multiplier
        self.LR_set_step = LR_set_step
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.kernel_sigma = kernel_sigma
        self.dist_threshold_scale = dist_threshold_scale
        self.kernel = None
        self.exp_scale = exp_scale
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
            if self.exp_scale:
                scale *= 1 + LR_set_step
            else:
                scale += LR_set_step
            scale = round(scale, 3)
        return result

    def _sum_part(self, image, intensity, replace_data, x, y):
        step_x, step_y, _ = replace_data.shape
        kernel = self.kernel
        if image[x:x + step_x, y:y + step_y].shape != replace_data.shape:
            step_x, step_y, _ = image[x:x + step_x, y:y + step_y].shape
            replace_data = get_subsample(replace_data, (0, step_x), (0, step_y))
            step_x, step_y, _ = replace_data.shape
            kernel = gkern(replace_data.shape[0], self.kernel_sigma)
            kernel = np.transpose(np.array([kernel,
                                            kernel,
                                            kernel]))

        image[x:x + step_x, y:y + step_y] += replace_data.astype("float32")
        intensity[x:x + step_x, y:y + step_y] += kernel

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
                continue

            upscale_coefficient = float(self.kernel.shape[0]) / replacement.shape[0] \
                if self.kernel is not None \
                else float(result_scale) / current_scale
            if current_scale != result_scale:
                replacement = upscale(
                    replacement, 
                    upscale_coefficient,
                )
            if self.kernel is None or replacement.shape != self.kernel.shape:
                s = replacement.shape[0]
                self.kernel = gkern(s, self.kernel_sigma)
                self.kernel = np.transpose(np.array([self.kernel,
                                                     self.kernel,
                                                     self.kernel]))
            if replacement.shape != self.kernel.shape:
                continue
            replacement = replacement * self.kernel
            x, y = patch["coords"]
            self._sum_part(result, intensity, replacement,
                           round(x * result_scale), round(y * result_scale))
            # show((255 * intensity / np.max(intensity)).astype("int8"))
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


        LR_patches = get_valid_patches(LR_patches)
        self.orig_patches = get_valid_patches(self.orig_patches)

        dataset = np.asarray([np.asarray(item["descriptor"])
                              for item in LR_patches])
        queryset = np.asarray([np.asarray(item["descriptor"])
                               for item in self.orig_patches])

        print(queryset.shape, dataset.shape)
        replace_to, dist = FLANN().nn(dataset, queryset, 1, algorithm="kmeans")
        threshold = np.median(dist) * self.dist_threshold_scale
        if not threshold:
            threshold = 10**10
        replace_to = [LR_patches[i] if dist[j] <= threshold else None for j, i in enumerate(replace_to)]

        print(len([item for item in replace_to if not item]), len(replace_to))
        for i, patch in enumerate(self.orig_patches):
            self.orig_patches[i]["replace_to"] = replace_to[i] if replace_to[i] else self.orig_patches[i]
        result, intensity_map = self._replace_parts(
            self.orig_patches, result_scale)
        upscaled = upscale(self.src_image, result_scale)

        # TODO: FIX THIS SHIT
        result[-2:-1, 0:-1] = upscaled[-2:-1, 0:-1]
        result[0:-1, -2:-1] = upscaled[0:-1, -2:-1]

        if is_show:
            intensity_map = (255 * intensity_map / np.max(intensity_map)).astype("int8")
            show([
                intensity_map,
            ])
        return result


def test_dataset_samples():
    # path = "../datasets/Urban100_SR/image_SRF_2/img_002_SRF_2_LR.png"
    # path = "../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png"
    path = "../datasets/Urban100_SR/image_SRF_2/img_013_SRF_2_LR.png"
    # path = "../datasets/Urban100_SR/image_SRF_2/img_006_SRF_2_LR.png"
    # path = "../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png"
    # path = '../datasets/BSD100_SR/image_SRF_2/img_011_SRF_2_LR.png'

    source = cv2.imread(path)
    scale = 2
    result = SuperResolutor(
        source,
        LR_set_step=0.25,
        downscale_multiplier=scale,
        patch_size=3,
        patch_step=1,
        kernel_sigma=0.1,
        dist_threshold_scale=4,
        exp_scale=False,
    ).scale(scale, False)

    # PSNR - 31.178822618057982
    # SSIM - 0.8058766721629134
    # IFC - 0.49936106189362844

    hr_path = path.replace('_LR', '_HR')
    hr_image = cv2.imread(hr_path)
    my = MetricAggregator.evaluate(result, hr_image)
    bicubic = MetricAggregator.evaluate(upscale(source, 2), hr_image)
    kim_path = path.replace('_LR', '_Kim')
    kim = MetricAggregator.evaluate(cv2.imread(kim_path), hr_image)
    glasner_path = path.replace('_LR', '_glasner')
    glasner = MetricAggregator.evaluate(cv2.imread(glasner_path), hr_image)
    s = 'my\n{}\n\nkim\n{}\nglasner\n{}\nbicubic\n{}'.format(my, kim, glasner, bicubic)
    print(s)
    # print('orig\n', MetricAggregator.evaluate(hr_image, hr_image))

    result_debug_path = '../results/debug/'
    filtered = cv2.bilateralFilter(result, 3, 90, 90)
    print('filtered', MetricAggregator.evaluate(filtered, hr_image))
    cv2.imwrite(os.path.join(result_debug_path, 'result.png'), result)
    cv2.imwrite(os.path.join(result_debug_path, 'result_filtered.png'), filtered)
    cv2.imwrite(os.path.join(result_debug_path, 'source.png'), source)
    cv2.imwrite(os.path.join(result_debug_path, 'result.png'), result)
    cv2.imwrite(os.path.join(result_debug_path, 'bicubic.png'), upscale(source, scale))
    cv2.imwrite(os.path.join(result_debug_path, 'hr_image.png'), hr_image)
    concat = get_concat((1 / 4, 3 / 4, 2 / 4, 3 / 4),
                        source,
                        cv2.imread(kim_path),
                        cv2.imread(glasner_path),
                        result,
                        upscale(source, scale),
                        )
    cv2.imwrite(os.path.join(result_debug_path, 'concat.png'), concat)
    with open(os.path.join(result_debug_path, 'metrics.txt'), 'w') as f:
        f.write(s)


def test_ellipse_samples():
    scale = 2
    # path = "../samples/black_ellipses.png"
    # path = "../samples/black_gradient_circle.jpeg"
    # path = "../samples/black_gradient_ellipse.jpeg"
    # path = "../samples/colored_gradient_ellipse.jpeg"
    path = "../samples/font_sizes.png"
    # path = "../samples/tag_cloud.jpg"
    hr = upscale(cv2.imread(path), 1)
    width, height, _ = hr.shape
    if width % scale:
        hr = hr[:width - width%scale]
    if height % scale:
        hr = hr[:,:height - height%scale]

    source = downscale(hr, scale)
    # source = downscale(hr, 1)

    result = SuperResolutor(
        source,
        LR_set_step=0.25,
        downscale_multiplier=scale,
        patch_size=3,
        patch_step=1,
        kernel_sigma=0.1,
        dist_threshold_scale=0,
        exp_scale=True,
    ).scale(scale, False)
    result_debug_path = '../results/debug/'
    bicubic = upscale(source, scale)
    downscaled = downscale(result, scale)
    filtered = cv2.bilateralFilter(result,3,90,90)
    cv2.imwrite(os.path.join(result_debug_path, 'result.png'), result)
    cv2.imwrite(os.path.join(result_debug_path, 'result_filtered.png'), filtered)
    cv2.imwrite(os.path.join(result_debug_path, 'downscaled.png'), downscaled)
    cv2.imwrite(os.path.join(result_debug_path, 'bicubic.png'), bicubic)
    cv2.imwrite(os.path.join(result_debug_path, 'hr_image.png'), hr)
    cv2.imwrite(os.path.join(result_debug_path, 'source.png'), source)

    concat = get_concat((2/4, 3/4, 0.2/4, 1.5/4), source, bicubic, result, hr)
    cv2.imwrite(os.path.join(result_debug_path, 'concat.png'), concat)

    my = MetricAggregator.evaluate(result, hr)
    bicubic = MetricAggregator.evaluate(bicubic, hr)
    s = 'my\n{}\nbicubic\n{}'.format(my, bicubic)
    print(s)
    print('filtered', MetricAggregator.evaluate(filtered, hr))
    with open(os.path.join(result_debug_path, 'metrics.txt'), 'w') as f:
        f.write(s)

    # show(np.abs(gray(result) - gray(hr)))

@timeit
def process(source):
    scale = 2
    result = SuperResolutor(
        source,
        LR_set_step=0.25,
        downscale_multiplier=scale,
        patch_size=3,
        patch_step=1,
        kernel_sigma=0.1,
        dist_threshold_scale=0,
        exp_scale=True,
    ).scale(scale, False)


def time_test():
    path = "../samples/colored_gradient_ellipse.jpeg"
    source = cv2.imread(path)
    source = downscale(source, 4)
    result = []
    for i in [item/4 for item in range(30)]:
        s = upscale(source, 1+i)
        _, t = process(s)
        t = int(t)/1000
        result.append((s.shape[0], s.shape[1], t))
    return result



if __name__ == '__main__':
    # test_dataset_samples()
    test_ellipse_samples()
    # r = time_test()
    # for item in r:
    #     print('{}\t{}\t{}'.format(*item))
    # print(1)


# PSNR - 32.322234988032385
# SSIM - 0.871911915380385
# IFC - 0.3507941796131255