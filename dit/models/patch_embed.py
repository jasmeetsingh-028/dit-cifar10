## Patchify the input image into non-overlapping patches and then flatten them.

import torch
import torch.nn as nn


class Patchify(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  ## 64 patches each of size 4*4*3

        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size = patch_size,
            stride = patch_size,  # no overlap between the patches
        )

    
    def forward(self, x):

        # x shape: (B, 3, 32, 32)
        x = self.proj(x) # (B, d_model, H/patch_size, W/patch_size) = (B, 384, 8, 8)
        x = x.flatten(2) # (B, 384, 8, 8) -> (B, 384, 64)
        x = x.transpose(1,2) # (B, 384, 64) -> (B, 384, 64) or (B, seq_len, d_model)
        return x
    

if __name__ == "__main__":
    patch_embed = Patchify()
    x = torch.randn(1, 3, 32, 32) # (B, c, h, w)
    out = patch_embed(x)
    print(out.shape) #shape: (B, 384, 64)
