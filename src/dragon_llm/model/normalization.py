import torch
from torch import nn


class DragonHeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-6):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, hidden_states):
        B, L, H, D = hidden_states.shape
        y = self.rms(hidden_states) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)


class DragonRMSNorm(nn.RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DragonRMSNorm is equivalent to RMSNorm
        """
        super().__init__(normalized_shape=hidden_size, eps=eps)