import torch.utils.data as data
from os.path import join, dirname, abspath
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np
import os

from utils import base_path


# base_path = dirname(abspath('train.py'))


def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class DatasetFromFolderVal(data.Dataset):
    def __init__(self, base_val_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        base_val_dir = join(base_path, base_val_dir)
        hr_dir = join(base_val_dir, 'HR')
        lr_dir = join(base_val_dir, 'LR', f'X{upscale}')

        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")
        if not os.path.exists(lr_dir):
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")


        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(img_modcrop(target, self.upscale))

        return input, target

    def __len__(self):
        return len(self.lr_filenames)
