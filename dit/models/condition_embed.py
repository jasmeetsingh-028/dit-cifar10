import torch
import math
import torch.nn as nn

def sinusoidal_embeddings(t, d_model): #d_model = 384

    """
    Create sinusoidal embeddings for the input tensor t.
    Args:
        t (torch.Tensor): Input tensor of shape (batch_size, d_model).
        d_model (int): Dimension of the model.
    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (batch_size, seq_len, d_model).
    """

    half = d_model // 2 #to divide the embedding dim into two halves for sine and cosine components

    # half = 192
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device = t.device) / (half - 1)  # 
    )

    #  exp(-log(10000) * (0, 1, ... 191) / 191 )
    # freq shape: (192,)
    args = t[:, None].float() * freqs[None]

    # t[:, None] shape: (B, 1) 
    # freqs[None] shape: (1, 192)
    # args shape: (B, half = 192)

    return torch.cat([torch.sin(args), torch.cos(args)], dim = -1) # (B, d_model = 384) = (B, half = 192) + (B, half = 192)


class TimeStepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, t):
        # t shape: (B,)
        t_emb = self.mlp(sinusoidal_embeddings(t, self.d_model)) # (B, d_model)


# classifier free guidance for class labels

# During training, 10% of the time model will not know the class label: replace y with a null token (index 10).

# So in a batch of 128 images, roughly 13 of them will have their class label silently replaced with the null token. The model never knows which ones
# model just sees index 10 and has to denoise without class information.

# why cfg? The model learns both conditional and unconditional denoising in one training run. 

