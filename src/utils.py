"""Shared helpers for training and evaluating the denoising model."""

import numpy as np
from jittor import init, nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def weights_init_kaiming(module):
    """Initialize trainable layers with Kaiming-style defaults."""

    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, a=0, mode="fan_in")
    elif isinstance(module, nn.BatchNorm2d):
        init.gauss_(module.weight, 1.0, 0.02)
        init.constant_(module.bias, 0.0)


def normalize_uint8(data):
    """Scale image data from [0, 255] to [0, 1]."""

    return data / 255.0


def batch_PSNR(img, clean, data_range):
    """Compute average PSNR for a batch of NCHW grayscale images."""

    restored = img.numpy().astype(np.float32)
    target = clean.numpy().astype(np.float32)
    score = 0.0
    for i in range(restored.shape[0]):
        score += peak_signal_noise_ratio(target[i], restored[i], data_range=data_range)
    return score / restored.shape[0]


def batch_SSIM(img, clean, data_range):
    """Compute average SSIM for a batch of NCHW grayscale images."""

    restored = img.numpy().astype(np.float32)
    target = clean.numpy().astype(np.float32)
    score = 0.0
    for i in range(restored.shape[0]):
        score += structural_similarity(
            target[i, 0],
            restored[i, 0],
            data_range=data_range,
        )
    return score / restored.shape[0]


def data_augmentation(image, mode):
    """Apply one of the eight DnCNN-style flip/rotation augmentations."""

    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        pass
    elif mode == 1:
        out = np.flipud(out)
    elif mode == 2:
        out = np.rot90(out)
    elif mode == 3:
        out = np.flipud(np.rot90(out))
    elif mode == 4:
        out = np.rot90(out, k=2)
    elif mode == 5:
        out = np.flipud(np.rot90(out, k=2))
    elif mode == 6:
        out = np.rot90(out, k=3)
    elif mode == 7:
        out = np.flipud(np.rot90(out, k=3))
    else:
        raise ValueError(f"Unsupported augmentation mode: {mode}")

    return np.ascontiguousarray(np.transpose(out, (2, 0, 1)))
