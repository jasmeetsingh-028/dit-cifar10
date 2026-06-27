import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloader(
    root="./data",
    batch_size=128,
    train=True,
    num_workers=4,
    download=True,
):
    """
    CIFAR-10 dataloader, images normalized to [-1, 1] to match the
    diffusion process's assumption of zero-centered data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                          # [0, 255] -> [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
    ])

    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,  # avoid uneven final batch during training
    )

    return loader


def denormalize(x):
    """Inverse of the Normalize transform: [-1, 1] -> [0, 1], for saving/viewing images."""
    return (x.clamp(-1, 1) + 1) / 2



if __name__ == "__main__":
    loader = get_cifar10_dataloader(batch_size=8, num_workers=0)

    x, y = next(iter(loader))
    print("batch x shape:", x.shape)   # ([8, 3, 32, 32])
    print("batch y shape:", y.shape)   # ([8])
    print("x min/max:", x.min().item(), x.max().item())  #  close to -1, 1
    print("y values:", y.tolist())     # class indices 0-9

    x_denorm = denormalize(x)

    print("x min/max:", x_denorm.min().item(), x_denorm.max().item()) # close to 0, 1