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

#class TimeStepEmbedding(nn.Module):

