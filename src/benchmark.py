from SuperResolutor import *

path = "../datasets/Set14/image_SRF_2/img_013_SRF_2_LR.png"
path = "../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png"
path = "../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png"

result_folder = "../results/"

def benchmark_one(path_LR, filename, patch_size, patch_step):
    scale = int(path_LR.split("/")[3].split("_")[-1])
    ext = path_LR.split(".")[-1]
    path_HR = path_LR.replace("_LR.", "_HR.")
    path_GL = path_LR.replace("_LR.", "_glasner.")
    LR_image = cv2.imread(path_LR)
    HR_image = cv2.imread(path_HR)
    GL_image = cv2.imread(path_GL) # glasner
    BC_image = upscale(LR_image, scale)
    result = SuperResolutor(LR_image,
        LR_set_step=0.5,
        downscale_multiplier=1.5,
        patch_size=patch_size,
        patch_step=patch_step).scale(scale)
    cv2.imwrite("%s%s_%s.%s" % (result_folder, filename, "galsner", ext), GL_image)
    cv2.imwrite("%s%s_%s.%s" % (result_folder, filename, "bicubic", ext), BC_image)
    cv2.imwrite("%s%s_%s.%s" % (result_folder, filename, "HR", ext), HR_image)
    cv2.imwrite("%s%s_%s_%d-%d.%s" % (result_folder, filename, "result", patch_size, patch_step, ext), result)

data = [
    # ("../datasets/Set5/image_SRF_2/img_002_SRF_2_LR.png", "bird"),
    # ("../datasets/Urban100_SR/image_SRF_2/img_001_SRF_2_LR.png", "building"),
    # ("../datasets/Set14/image_SRF_2/img_013_SRF_2_LR.png", "scan"),
    ("../datasets/Urban100_SR/image_SRF_2/img_006_SRF_2_LR.png", "panel"),
]

for path, name in data:
    for size, step in [(4, 2), (2, 1), (4, 1),]:
        print(name, size, step)
        benchmark_one(path, name, size, step)

