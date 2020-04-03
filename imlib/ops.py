import numpy as np


def cal_grid_index(i, unit, margin):
    start_index = i * unit + (i + 1) * margin
    end_index = (i + 1) * unit + (i + 1) * margin
    return start_index, end_index


def mk_grid(imgs,  margin=5):
    if not isinstance(imgs, list):
        imgs = [imgs]

    h, w, c = imgs[0].shape
    num_cimgs = np.ceil(np.sqrt(len(imgs))).astype(int)
    num_rimgs = np.ceil(len(imgs) / num_cimgs).astype(int)
    canvas = np.zeros((num_rimgs * h + (num_rimgs + 1) * margin, (num_cimgs * w + (num_cimgs + 1) * margin), c),
                      dtype=np.uint8)

    for i in range(len(imgs)):
        r_i, c_i = divmod(i, num_cimgs)
        str_h, end_h = cal_grid_index(r_i, h, margin)
        str_w, end_w = cal_grid_index(c_i, w, margin)
        canvas[str_h:end_h, str_w:end_w, :] = imgs[i]

    return canvas