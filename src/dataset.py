"""Dataset preparation and HDF5-backed data loading."""

import glob
import os
import random

import cv2
import h5py
import jittor as jt
import numpy as np
from jittor.dataset import Dataset as JtDataset

from src.utils import data_augmentation, normalize_uint8


DEFAULT_SCALES = (1.0, 0.9, 0.8, 0.7)


def crop_to_multiple(image, factor=16):
    """Crop image height and width so they are divisible by ``factor``."""

    if image.ndim == 3 and image.shape[0] == 1:
        height, width = image.shape[1], image.shape[2]
        chw = True
    else:
        height, width = image.shape[0], image.shape[1]
        chw = False

    new_height = (height // factor) * factor
    new_width = (width // factor) * factor
    if new_height == 0 or new_width == 0:
        raise ValueError(
            f"Image size {height}x{width} is too small after cropping to {factor}."
        )

    if chw:
        return image[:, :new_height, :new_width]
    return image[:new_height, :new_width, ...]


def image_to_patches(image, patch_size, stride=1):
    """Extract sliding-window patches from a CHW image."""

    from numpy.lib.stride_tricks import as_strided

    channels, height, width = image.shape
    out_height = (height - patch_size) // stride + 1
    out_width = (width - patch_size) // stride + 1
    channel_stride, height_stride, width_stride = image.strides
    patches = as_strided(
        image,
        shape=(channels, out_height, out_width, patch_size, patch_size),
        strides=(
            channel_stride,
            height_stride * stride,
            width_stride * stride,
            height_stride,
            width_stride,
        ),
    )
    return patches.transpose(0, 3, 4, 1, 2).reshape(channels, patch_size, patch_size, -1)


# Backward-compatible name used by the original script.
Im2Patch = image_to_patches


def load_grayscale_chw(path):
    """Read an image as normalized grayscale CHW data."""

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = np.expand_dims(image, axis=0)
    return np.float32(normalize_uint8(image))


def prepare_data(
    data_path,
    patch_size,
    stride,
    aug_times=1,
    train_output="train.h5",
    val_output="val.h5",
):
    """Build train/validation HDF5 files from ``data/train`` and ``data/Set12``."""

    train_files = sorted(glob.glob(os.path.join(data_path, "train", "*.png")))
    if not train_files:
        raise FileNotFoundError(f"No training images found under {data_path}/train")

    print("Processing training data")
    train_num = 0
    with h5py.File(train_output, "w") as h5f:
        for file_path in train_files:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {file_path}")

            for scale in DEFAULT_SCALES:
                resized = cv2.resize(
                    image,
                    (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
                resized = crop_to_multiple(resized, factor=16)
                chw_image = np.float32(normalize_uint8(np.expand_dims(resized, axis=0)))
                patches = image_to_patches(chw_image, patch_size=patch_size, stride=stride)
                print(
                    "file: %s scale %.1f # samples: %d"
                    % (file_path, scale, patches.shape[3] * aug_times)
                )

                for patch_idx in range(patches.shape[3]):
                    patch = patches[:, :, :, patch_idx].copy()
                    h5f.create_dataset(str(train_num), data=patch)
                    train_num += 1

                    for aug_idx in range(aug_times - 1):
                        augmented = data_augmentation(patch, random.randint(1, 7))
                        h5f.create_dataset(f"{train_num}_aug_{aug_idx + 1}", data=augmented)
                        train_num += 1

    print("\nProcessing validation data")
    val_files = sorted(glob.glob(os.path.join(data_path, "Set12", "*.png")))
    if not val_files:
        raise FileNotFoundError(f"No validation images found under {data_path}/Set12")

    val_num = 0
    with h5py.File(val_output, "w") as h5f:
        for file_path in val_files:
            print(f"file: {file_path}")
            image = crop_to_multiple(load_grayscale_chw(file_path), factor=16)
            h5f.create_dataset(str(val_num), data=image)
            val_num += 1

    print(f"training set, # samples {train_num}\n")
    print(f"validation set, # samples {val_num}\n")


class Dataset(JtDataset):
    """Lazy HDF5 dataset for Jittor DataLoader workers."""

    def __init__(self, train=True, train_path="train.h5", val_path="val.h5"):
        super().__init__()
        self.train = train
        self.h5_path = train_path if train else val_path
        self.h5f = None

        with h5py.File(self.h5_path, "r") as h5f:
            self.keys = list(h5f.keys())
        if self.train:
            random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, "r")
        return jt.array(np.array(self.h5f[self.keys[index]]))
