from SuperResolutor import *
from Metrics import PSNR
import json
from collections import OrderedDict
import os

path = "../datasets/Set14/image_SRF_2/img_013_SRF_2_LR.png"
path = "../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png"
path = "../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png"

result_folder = "../results/"


def benchmark_one(path_LR, filename):
    scores = []
    for patch_size in range(2, 7, 1):#[2]:
        for set_step in [round(item/100.0, 3) for item in range(5, 55, 5)]:#[0.5]:
            for sigma in [round(item/10.0, 3) for item in range(1, 31, 1)]:#[1,2]:
                print(name, patch_size, set_step, sigma)
                scale = int(path_LR.split("/")[3].split("_")[-1])
                ext = path_LR.split(".")[-1]
                path_HR = path_LR.replace("_LR.", "_HR.")
                path_GL = path_LR.replace("_LR.", "_glasner.")
                LR_image = cv2.imread(path_LR)
                HR_image = cv2.imread(path_HR)
                GL_image = cv2.imread(path_GL)  # glasner
                BC_image = upscale(LR_image, scale)
                result = SuperResolutor(LR_image,
                                        LR_set_step=set_step,
                                        downscale_multiplier=50,
                                        patch_size=patch_size,
                                        patch_step=1,
                                        kernel_sigma=sigma).scale(scale)
                cv2.imwrite("%s%s/%s_%.3f.%s" %
                            (result_folder, filename, "galsner", PSNR.evaluate(HR_image, GL_image), ext), GL_image)
                cv2.imwrite("%s%s/%s_%.3f.%s" %
                            (result_folder, filename, "bicubic", PSNR.evaluate(HR_image, BC_image), ext), BC_image)
                cv2.imwrite("%s%s/%s.%s" % (result_folder, filename, "HR", ext), HR_image)
                score = PSNR.evaluate(HR_image, result)
                rfname = "%s%s/%d-%.3f-%.1f_%.3f.%s" % (result_folder, filename,
                                                        patch_size, set_step, sigma,
                                                        score, ext)
                scores.append(OrderedDict([
                    ("name", rfname),
                    ("score", score),
                    ("set_step", set_step),
                    ("sigma", sigma),
                    ("patch_size", patch_size)
                ]))
                if not os.path.exists(result_folder + filename):
                    os.makedirs(result_folder + filename)
                cv2.imwrite(rfname, result)
    scores = sorted(scores, key=lambda item: item["score"], reverse=True)
    with open("{0}{1}/scores.json".format(result_folder, filename), "w") as f:
        f.write(json.dumps(scores, indent=2))

data = [
    # ("../datasets/Set5/image_SRF_2/img_001_SRF_2_LR.png"),
    ("../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png"),
    # ("../datasets/Set5/image_SRF_2/img_003_SRF_2_LR.png"),
    # ("../datasets/Set5/image_SRF_2/img_004_SRF_2_LR.png"),
    # ("../datasets/Set5/image_SRF_2/img_005_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_002_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_003_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_004_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_005_SRF_2_LR.png"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_006_SRF_2_LR.png"),
    # ("../datasets/Set14/image_SRF_2/img_001_SRF_2_LR.png"),
    # ("../datasets/Set14/image_SRF_2/img_002_SRF_2_LR.png"),
    # ("../datasets/Set14/image_SRF_2/img_003_SRF_2_LR.png"),
    # ("../datasets/Set14/image_SRF_2/img_004_SRF_2_LR.png"),
    # ("../datasets/Set14/image_SRF_2/img_013_SRF_2_LR.png"),
]

for i, path in enumerate(data):
    name = "result_" + str(i)
    benchmark_one(path, name)
