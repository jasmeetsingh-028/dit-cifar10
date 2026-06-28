import torch
from dit.data.cifar10 import denormalize


def save_checkpoint(path, model, ema, optimizer, step, epoch):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }, path)


def load_checkpoint(path, model, ema, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["epoch"]


def save_sample_grid(images, path, nrow=4):
    # images: (B, 3, 32, 32) in [-1, 1]
    from torchvision.utils import save_image
    images = denormalize(images)  # -> [0, 1]
    save_image(images, path, nrow=nrow)