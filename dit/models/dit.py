import torch
import torch.nn as nn

from dit_block import DITBlock
from patch_embed import Patchify
from condition_embed import ConditionEmbedding

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
        self.conditional_embed = clas
