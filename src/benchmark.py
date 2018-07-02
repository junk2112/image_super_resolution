from src.SuperResolutor import *


base = '/Users/andrew/git/image_super_resolution/datasets'

def benchmark(args):
    i, path = args
    print(i + 1, len(source_patches))
    hr_path = path.replace('_LR', '_HR')
    # kim_path = path.replace('_LR', '_Kim')
    # glasner_path = path.replace('_LR', '_glasner')
    # glasner_path = path.replace('_LR', '_our')

    source = cv2.imread(path)
    hr = cv2.imread(hr_path)
    # kim = cv2.imread(kim_path)
    # glasner = cv2.imread(glasner_path)

    # result = SuperResolutor(
    #     source,
    #     LR_set_step=0.25,
    #     downscale_multiplier=scale,
    #     patch_size=3,
    #     patch_step=1,
    #     kernel_sigma=0.1,
    #     dist_threshold_scale=4,
    #     exp_scale=True,
    # ).scale(scale, False)

    return {
        # 'kim': MetricAggregator.evaluate(kim, hr),
        # 'glasner': MetricAggregator.evaluate(glasner, hr),
        # 'my': MetricAggregator.evaluate(result, hr),
        'bicubic': MetricAggregator.evaluate(upscale(source, scale), hr),
    }
# set_name = 'Urban100_SR'
# set_name = 'BSD100_SR'
set_name = 'SunHays80_SR'
# set_name = 'Set5'
# set_name = 'Set14'
scales = [2, 3, 4, 8]
printings = []

for scale in scales:
    data_path = base + '/{}/image_SRF_{}'.format(set_name, scale)
    try:
        source_patches = list_files(data_path, lambda item: '_LR' in item)
    except:
        continue

    metrics = []
    for i, path in enumerate(source_patches):
        metrics.append(benchmark((i, path)))

    # for method_name in ['kim', 'glasner', 'my', 'bicubic']:
    for method_name in ['bicubic']:
        method = [item[method_name] for item in metrics]
        for metric_name in ['PSNR', 'SSIM', 'VIF']:
            metric = [item[metric_name] for item in method]
            printings.append((
                set_name,
                ', Scale: {}, '.format(scale),
                method_name,
                metric_name, '%.3f' % (sum(metric)/len(metric))
            ))

for p in printings:
    print(*p)
