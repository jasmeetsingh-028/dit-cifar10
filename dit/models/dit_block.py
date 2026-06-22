import torch
import torch.nn as nn
from dit_utils import Attention, FFN, modulate

class DITBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model, elementwise_affine = True)
        self.attn = Attention(d_model = d_model, num_heads = num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = FFN(d_model = d_model, mlp_ratio = mlp_ratio)

        # Adaptive layer norm
        # produce gamma1, beta1, alpha1, gamma2, beta2, alpha2 in one shot

        self.adaLN_parameters = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )

        # initialize weights and bias of adaLN with zeros
        nn.init.zeros_(self.adaLN_parameters[-1].weight)
        nn.init.zeros_(self.adaLN_parameters[-1].bias)
    
    def forward(self, x, c):
        # x shape: (B, T: seq_len, C: d_model)
        # c shape: (B, c: d_model)

        params = self.adaLN_parameters(c)  # shape: (B, 6 * d_model)
        # chunk params along the last dim into 6 params
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        #shapes remains contant throughout

        x_norm = modulate(self.layer_norm1(x), gamma1, beta1)
        x = x + alpha1.unsqueeze(1) * self.attn(x_norm)

        x_norm = modulate(self.layer_norm2(x), gamma2, beta2)
        x = x + alpha2.unsqueeze(1) * self.ffn(x_norm)
        # x shape: (B, T: seq_len, C: d_model)

        return x
    
if __name__ == "__main__":

    x = torch.randn(1, 64, 512)
    c = torch.randn(1, 512)
    dit = DITBlock(d_model = 512, num_heads = 8, mlp_ratio = 4.0)
    out = dit(x, c)
    print(out.shape)


        