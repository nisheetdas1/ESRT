import torch.utils.data as data
from os.path import join, dirname, abspath, exists
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np
import os

# Attempt to get base_path dynamically, but avoid using it for dataset paths
try:
    from utils import base_path
except ImportError:
    # Fallback if utils or base_path is not available
    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Warning: Could not import base_path from utils. Using fallback: {base_path}")


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
        # base_val_dir = join(base_path, base_val_dir) # <--- Remove this line

        # Construct absolute paths based on the provided base_val_dir
        # Assuming base_val_dir is relative to the execution directory of train.py
        abs_base_val_dir = os.path.abspath(base_val_dir)
        hr_dir = join(abs_base_val_dir, 'HR')
        lr_dir = join(abs_base_val_dir, 'LR', f'X{upscale}')

        print(f"Looking for HR images in: {hr_dir}") # Debug print
        print(f"Looking for LR images in: {lr_dir}") # Debug print

        if not exists(hr_dir): # Use exists from os.path
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")
        if not exists(lr_dir): # Use exists from os.path
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")


        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input_img = load_image(self.lr_filenames[index])
        target_img = load_image(self.hr_filenames[index])

        # Apply modcrop before converting to tensor
        target_img = img_modcrop(target_img, self.upscale)

        input_tensor = np2tensor()(input_img)
        target_tensor = np2tensor()(target_img)


        return input_tensor, target_tensor

    def __len__(self):
        # Ensure length is consistent between HR and LR lists
        if len(self.lr_filenames) != len(self.hr_filenames):
            print(f"Warning: Mismatch in number of LR ({len(self.lr_filenames)}) and HR ({len(self.hr_filenames)}) files.")
            # Optionally, raise an error or take the minimum length
            # raise ValueError("Number of LR and HR files must match")
            return min(len(self.lr_filenames), len(self.hr_filenames))
        return len(self.lr_filenames)
