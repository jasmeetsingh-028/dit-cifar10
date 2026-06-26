import torch
import torch.nn as nn

from dit.models.dit_block import DITBlock
from dit.models.patch_embed import Patchify
from dit.models.condition_embed import ConditionEmbedding

class DIT(nn.Module):
    def __init__(self, img_size = 32,
                 patch_size = 4,
                 in_channels = 3,
                 d_model = 384,
                 n_blocks = 12,
                 num_heads = 6,
                 mlp_ratio = 4.0,
                 num_classes = 10,
                 class_dropout_p = 0.1):
        
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels # preducting noise same as the input image shape

        self.patch_embed = Patchify(img_size = img_size,
                                    patch_size = patch_size,
                                    in_channels = in_channels,
                                    d_model = d_model)
        
        num_patches = self.patch_embed.num_patches

        # learned positional embeddings

        # image patches -> positonal embedding to get idea of postion of each patch in the input image

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))  # learnable positional emdedding initialization, creates a tensor of shape: (1, num_patches, d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # initialize pos embeddings with mean and 

        # conditional embedding for class and timestep
        self.conditional_embed = ConditionEmbedding(
            d_model = d_model,
            num_classes = num_classes,
            dropout_p = class_dropout_p
        )

        # DIT blocks
        self.dit_blocks = nn.ModuleList(
            [
                DITBlock(
                    d_model, num_heads, mlp_ratio
                ) for _ in range(n_blocks)
            ]
        )

        # final adaln layer and linear layer
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)  # only scaling and shifting, no gate needed here
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        patch_dim = patch_size * patch_size * self.out_channels   # 4 * 4 * 3 = 48

        self.final_linear_proj = nn.Linear(d_model, patch_dim) # to unpatchify 384 -> 48

        # zero initilaization for final layer

        nn.init.zeros_(self.final_linear_proj.weight)
        nn.init.zeros_(self.final_linear_proj.bias)
    
    def unpatchify(self, x):
        # x shape after dit: (B, seq_len/num_patches = 64, 4*4*3 = 48) where patch_Size/patch height and width = 4 and patch channels = 3
        B, T, _ = x.shape # T = 64
        P = self.patch_size  # 4
        H = W = int(T ** 0.5) # sqrt(64) =  8
        C =  self.out_channels #3

        x = x.reshape(B, H, W, P, P, C) #(B, 8, 8, 4, 4, 3)
        x = x.permute(0, 5, 1, 3, 2, 4) # (B, C = 3, H = 8, P = 4, W = 8, P = 4)
        x = x.reshape(B, C, H*P, W*P) #(B, 3, 32, 32)
        return x
    
    def forward(self, x, t, y):

        # x: (B, 3, 32, 32) noisy image
        # t: (B,) timesteps
        # y: (B,) class labels

        x = self.patch_embed(x) + self.pos_embed  # patchify image and add positional embeddings to the image
        # x shape: (B, seq_len/num patches = 64, d_model = 384)
        c = self.conditional_embed(t, y)
        # c shape: (B, d_model = 384)

        for block in self.dit_blocks:
            x = block(x, c)

        #x shape: (B, seq_len/num patches = 64, d_model = 384)

        # final adaLN (scale + shift only, no gate)
        gamma, beta = self.final_adaLN(c).chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        x = self.final_linear_proj(x)

        # x shape: (B, seq_len/num patches = 64, d_model = 384) -> (B, num_patches/seq_len/T = 64, patch_dim = 48 = 4*4*3)
        x = self.unpatchify(x)

        return x
    

if __name__ == "__main__":
    model = DIT()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))

    out = model(x, t, y)
    print(out.shape)   # expect: torch.Size([4, 3, 32, 32])
    assert out.shape == x.shape

    