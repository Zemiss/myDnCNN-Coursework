"""Train the blind grayscale denoising model."""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_nested, load_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            stream.flush()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _build_namespace(config):
    return Namespace(
        preprocess=get_nested(config, "train.preprocess"),
        batch_size=get_nested(config, "train.batch_size"),
        num_of_layers=get_nested(config, "model.num_of_layers"),
        epochs=get_nested(config, "train.epochs"),
        milestone=get_nested(config, "train.milestone"),
        lr=get_nested(config, "train.lr"),
        mode=get_nested(config, "train.mode"),
        noiseL=get_nested(config, "train.noiseL"),
        val_noiseL=get_nested(config, "train.val_noiseL"),
        data_dir=get_nested(config, "paths.data_dir"),
        train_h5=get_nested(config, "paths.train_h5"),
        val_h5=get_nested(config, "paths.val_h5"),
        num_workers=get_nested(config, "train.num_workers"),
        use_cuda=get_nested(config, "train.use_cuda"),
        patch_size=get_nested(config, "preprocess.patch_size"),
        stride=get_nested(config, "preprocess.stride"),
        model_dir=get_nested(config, "paths.model_dir"),
        checkpoint_name=get_nested(config, "paths.checkpoint_name"),
        log_dir=get_nested(config, "paths.log_dir"),
        log_file=get_nested(config, "paths.log_file"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DnCNN-style blind denoiser")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--preprocess", type=str2bool, default=None)
    parser.add_argument("--batch-size", "--batchSize", dest="batch_size", type=int, default=None)
    parser.add_argument("--num_of_layers", type=int, default=None, help="Kept for old commands.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--milestone", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--outf", dest="log_dir", type=str, default=None, help="Legacy alias for --log-dir.")
    parser.add_argument("--log-dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--log-file", dest="log_file", type=str, default=None)
    parser.add_argument("--model-dir", dest="model_dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", dest="checkpoint_name", type=str, default=None)
    parser.add_argument("--mode", choices=("S", "B"), default=None)
    parser.add_argument("--noiseL", type=float, default=None, help="Noise level for mode S.")
    parser.add_argument("--val_noiseL", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-h5", dest="train_h5", type=str, default=None)
    parser.add_argument("--val-h5", dest="val_h5", type=str, default=None)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-cuda", type=str2bool, default=None)
    args = parser.parse_args(argv)
    overrides = {
        "train.preprocess": args.preprocess,
        "train.batch_size": args.batch_size,
        "model.num_of_layers": args.num_of_layers,
        "train.epochs": args.epochs,
        "train.milestone": args.milestone,
        "train.lr": args.lr,
        "paths.log_dir": args.log_dir,
        "paths.log_file": args.log_file,
        "paths.model_dir": args.model_dir,
        "paths.checkpoint_name": args.checkpoint_name,
        "train.mode": args.mode,
        "train.noiseL": args.noiseL,
        "train.val_noiseL": args.val_noiseL,
        "paths.data_dir": args.data_dir,
        "paths.train_h5": args.train_h5,
        "paths.val_h5": args.val_h5,
        "preprocess.patch_size": args.patch_size,
        "preprocess.stride": args.stride,
        "train.num_workers": args.num_workers,
        "train.use_cuda": args.use_cuda,
    }
    return _build_namespace(load_config(args.config, overrides))


def make_noise(clean, mode, noise_level, blind_range=(0, 55)):
    """Create Gaussian noise and the matching noise-level map."""

    import jittor as jt
    import numpy as np

    if mode == "S":
        sigma = noise_level / 255.0
        return jt.randn(clean.shape) * sigma, jt.full_like(clean, sigma)

    noise = jt.zeros(clean.shape)
    noise_map = jt.zeros(clean.shape)
    sigmas = np.random.uniform(blind_range[0], blind_range[1], size=clean.shape[0])
    for batch_idx, sigma in enumerate(sigmas):
        sigma = sigma / 255.0
        sample_shape = noise[batch_idx, :, :, :].shape
        noise[batch_idx, :, :, :] = jt.randn(sample_shape) * sigma
        noise_map[batch_idx, :, :, :] = sigma
    return noise, noise_map


def build_loaders(args):
    from jittor.dataset import DataLoader

    from src.dataset import Dataset

    train_set = Dataset(train=True, train_path=args.train_h5, val_path=args.val_h5)
    val_set = Dataset(train=False, train_path=args.train_h5, val_path=args.val_h5)
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader


def train_one_epoch(model, criterion, optimizer, loader, args, epoch, step, writer, train_size):
    import jittor as jt

    from src.utils import batch_PSNR, batch_SSIM

    num_batches = (train_size + args.batch_size - 1) // args.batch_size
    current_lr = args.lr if epoch < args.milestone else args.lr / 10.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    print(f"learning rate {current_lr:.6f}")

    model.train()
    for batch_idx, clean in enumerate(loader, start=1):
        optimizer.zero_grad()
        noise, noise_map = make_noise(clean, args.mode, args.noiseL)
        noisy = clean + noise
        model_input = jt.concat([noisy, noise_map], dim=1)

        predicted_noise = model(model_input)
        loss = criterion(predicted_noise, noise) / (noisy.shape[0] * 2)
        optimizer.step(loss)

        restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)
        psnr = batch_PSNR(restored, clean, 1.0)
        ssim = batch_SSIM(restored, clean, 1.0)
        print(
            "[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f"
            % (epoch + 1, batch_idx, num_batches, loss.item(), psnr, ssim)
        )

        if step % 10 == 0:
            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("PSNR/train", psnr, step)
            writer.add_scalar("SSIM/train", ssim, step)
        step += 1
    return step


def validate(model, loader, args, val_size):
    import jittor as jt

    from src.utils import batch_PSNR, batch_SSIM

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    with jt.no_grad():
        for clean in loader:
            sigma = args.val_noiseL / 255.0
            noise = jt.randn(clean.shape) * sigma
            noisy = clean + noise
            noise_map = jt.full_like(clean, sigma)
            model_input = jt.concat([noisy, noise_map], dim=1)

            predicted_noise = model(model_input)
            restored = jt.clamp(noisy - predicted_noise, 0.0, 1.0)
            total_psnr += batch_PSNR(restored, clean, 1.0)
            total_ssim += batch_SSIM(restored, clean, 1.0)

    return total_psnr / val_size, total_ssim / val_size


def main():
    import jittor as jt
    from jittor import nn, optim
    from tensorboardX import SummaryWriter

    from src.dataset import prepare_data
    from src.models import UNet
    from src.utils import weights_init_kaiming

    args = parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_handle = open(args.log_file, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)

    if args.preprocess:
        aug_times = 1 if args.mode == "S" else 2
        prepare_data(
            data_path=args.data_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            aug_times=aug_times,
            train_output=args.train_h5,
            val_output=args.val_h5,
        )

    print("Loading dataset ...")
    train_set, val_set, train_loader, val_loader = build_loaders(args)
    print(f"# of training samples: {len(train_set)}")

    model = UNet(channels=1)
    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    step = 0
    for epoch in range(args.epochs):
        step = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            args,
            epoch,
            step,
            writer,
            len(train_set),
        )
        psnr, ssim = validate(model, val_loader, args, len(val_set))
        print(f"\n[epoch {epoch + 1}] PSNR_val: {psnr:.4f} SSIM_val: {ssim:.4f}")
        writer.add_scalar("PSNR/val", psnr, epoch)
        writer.add_scalar("SSIM/val", ssim, epoch)
        jt.save(model.state_dict(), os.path.join(args.model_dir, args.checkpoint_name))

    writer.close()


if __name__ == "__main__":
    main()
