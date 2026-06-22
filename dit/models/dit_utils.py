import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0 #to split q, k, v into num_heads
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # dim for each q, k, v which are split into n_heads

        self.qkv = nn.Linear(d_model, 3* d_model)
        self.proj = nn.Linear(d_model, d_model)

    
    def forward(self, x):

        B, T, C = x.shape #batch_size, seq_length, embedding dim

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim) #(B, seq_len, num_heads, head_dim) after splitting q, k, v into n_heads
        qkv = qkv.permute(2, 0, 3, 1, 4)  #(shape: (3, B, num_heads, seq_len (T), self.head_dim))

        q, k, v = qkv[0], qkv[1], qkv[2] #shape: (B, num_heads, seq_len, head_dim)

        attn = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5) # output attention scores
        #shape: (B, num_heads, seq_len, head_dim) @ (B, num_heads, head_dim, seq_len) (B, num_heads, seq_len, seq_len)
        attn = attn.softmax(dim = -1)  # ouput probabilities

        out = attn @ v #shape(B, num_heads, seq_len T, head_dim)

        out = out.transpose(1, 2).reshape(B, T, C)
        # Shape: (Batch_size, seq_len, embed_dim)

        return self.proj(out)

class FFN(nn.Module):
    def __init__(self, d_model, mlp_ratio = 4.0):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, d_model)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)



    
if __name__ == '__main__':

    x = torch.randn(1, 64, 512)
    attention = Attention(d_model = 512, num_heads=8)
    out = attention(x)
    print(out.shape)

