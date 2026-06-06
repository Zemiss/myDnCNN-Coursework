"""Evaluate a trained denoising model on Set12 or Set68."""

import argparse
import glob
import os
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_nested, load_config


PAD_FACTOR = 8
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def _build_namespace(config):
    return Namespace(
        num_of_layers=get_nested(config, "model.num_of_layers"),
        model_dir=get_nested(config, "paths.model_dir"),
        checkpoint_name=get_nested(config, "paths.checkpoint_name"),
        test_data=get_nested(config, "eval.test_data"),
        test_noiseL=get_nested(config, "eval.test_noiseL"),
        data_dir=get_nested(config, "paths.data_dir"),
        use_cuda=get_nested(config, "train.use_cuda"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DnCNN-style denoiser")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--num_of_layers", type=int, default=None, help="Kept for old commands.")
    parser.add_argument("--logdir", dest="model_dir", type=str, default=None, help="Legacy alias for --model-dir.")
    parser.add_argument("--model-dir", dest="model_dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", dest="checkpoint_name", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None, choices=("Set12", "Set68"))
    parser.add_argument("--test_noiseL", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--use-cuda", type=str2bool, default=None)
    args = parser.parse_args(argv)
    overrides = {
        "model.num_of_layers": args.num_of_layers,
        "paths.model_dir": args.model_dir,
        "paths.checkpoint_name": args.checkpoint_name,
        "eval.test_data": args.test_data,
        "eval.test_noiseL": args.test_noiseL,
        "paths.data_dir": args.data_dir,
        "train.use_cuda": args.use_cuda,
    }
    return _build_namespace(load_config(args.config, overrides))


def pad_to_multiple(tensor, factor=PAD_FACTOR):
    """Pad BCHW tensor on the bottom/right edges to match the U-Net stride."""

    import jittor as jt
    import numpy as np

    _, _, height, width = tensor.shape
    padded_height = (height + factor - 1) // factor * factor
    padded_width = (width + factor - 1) // factor * factor
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width

    padded = np.pad(tensor.numpy(), ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return jt.array(padded), height, width


def load_image(path):
    import cv2
    import jittor as jt
    import numpy as np

    from src.utils import normalize_uint8

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = normalize_uint8(np.float32(image))
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    return jt.array(image)


def evaluate_image(model, clean, noise_level):
    import jittor as jt

    from src.utils import batch_PSNR, batch_SSIM

    padded_clean, original_height, original_width = pad_to_multiple(clean)
    sigma = noise_level / 255.0
    noise = jt.randn(padded_clean.shape) * sigma
    noisy = padded_clean + noise
    noise_map = jt.full_like(padded_clean, sigma)
    model_input = jt.concat([noisy, noise_map], dim=1)

    with jt.no_grad():
        predicted_noise = model(model_input)
        restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)

    restored = restored[..., :original_height, :original_width]
    return batch_PSNR(restored, clean, 1.0), batch_SSIM(restored, clean, 1.0)


def main():
    import jittor as jt

    from src.models import UNet

    args = parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    print(f"Using Jittor with CUDA: {jt.flags.use_cuda}")

    model_path = os.path.join(args.model_dir, args.checkpoint_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading model ...")
    model = UNet(channels=1)
    model.load_state_dict(jt.load(model_path))
    model.eval()

    pattern = os.path.join(args.data_dir, args.test_data, "*.png")
    image_files = sorted(glob.glob(pattern))
    if not image_files:
        raise FileNotFoundError(f"No images found: {pattern}")

    total_psnr = 0.0
    total_ssim = 0.0
    for image_path in image_files:
        clean = load_image(image_path)
        psnr, ssim = evaluate_image(model, clean, args.test_noiseL)
        total_psnr += psnr
        total_ssim += ssim
        print(f"{os.path.basename(image_path)} PSNR {psnr:.4f} SSIM {ssim:.4f}")

    count = len(image_files)
    print(f"\nAverage PSNR on {args.test_data} (Noise={args.test_noiseL}): {total_psnr / count:.4f}")
    print(f"Average SSIM on {args.test_data} (Noise={args.test_noiseL}): {total_ssim / count:.4f}")


if __name__ == "__main__":
    main()
