import os
import cv2
import numpy as np

SAVE_DIR = './save/video'


def all_files_under(path, extension=None, append_path=True, sort=True):
    if not isinstance(extension, list):
        extension = list(extension)

    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = list()
            for ext in extension:
                filenames += [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(ext)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = list()
            for ext in extension:
                filenames += [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(ext)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def ImgReader(path, ext, resize_factor=None, interpolation=cv2.INTER_CUBIC):
    img_paths = all_files_under(path, ext)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if resize_factor is not None:
            img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=interpolation)

        yield os.path.basename(img_path), img


def mk_grid(imgs,  margin=5):
    if not isinstance(imgs, list):
        imgs = [imgs]

    h, w, c = imgs[0].shape
    num_cimgs = np.ceil(np.sqrt(len(imgs))).astype(int)
    num_rimgs = np.ceil(len(imgs) / num_cimgs).astype(int)

    canvas = np.zeros((num_cimgs * h + (num_cimgs + 1) * margin, (num_rimgs * w + (num_rimgs + 1) * margin)),
                      dtype=np.uint8)
    for i in range(len(imgs)):
        r_i = i // num_cimgs
        c_i = i % num_cimgs

        str_h = r_i * h + (r_i + 1) * margin
        end_h = (r_i + 1) * h + (r_i + 1) * margin
        str_w = c_i * w + (c_i + 1) * margin
        end_w = (c_i + 1) * w + (c_i + 1) * margin
        canvas[str_h: end_h, str_w: end_w , :] = imgs[i]