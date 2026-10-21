import torch
from torch import nn
import torch.nn.functional as F
import math


class _ScaledLinearFB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, alpha_fwd, alpha_bwd_x, alpha_bwd_w):
        ctx.save_for_backward(x, weight, bias)
        ctx.alpha_bwd_x = alpha_bwd_x
        ctx.alpha_bwd_w = alpha_bwd_w
        return F.linear(x, weight, bias) * alpha_fwd

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        # -------- grads ----------
        grad_x = torch.matmul(grad_out * ctx.alpha_bwd_x, weight)

        go_flat = (grad_out * ctx.alpha_bwd_w).reshape(-1, grad_out.shape[-1])
        x_flat  = x.reshape(-1, x.shape[-1])
        grad_weight = go_flat.t() @ x_flat
        grad_bias   = go_flat.sum(0) if bias is not None else None

        return grad_x, grad_weight, grad_bias, None, None, None


class DragonLinear(nn.Linear):
    """Linear layer with different forward/backward scalings."""
    def __init__(self, use_uscaling: bool, in_features, out_features, bias=False, alpha_fwd=None, alpha_bwd=None):
        super().__init__(in_features, out_features, bias)

        if alpha_fwd is None:
            alpha_fwd = 1.0 / math.sqrt(in_features)

        if not use_uscaling:
            alpha_fwd, alpha_bwd = 1, 1

        self.register_buffer("alpha_fwd", torch.tensor(float(alpha_fwd)), persistent=False)
        self.register_buffer("alpha_bwd", torch.tensor(float(alpha_bwd if alpha_bwd is not None else alpha_fwd)), persistent=False)

    def forward(self, x):
        return _ScaledLinearFB.apply(x, self.weight, self.bias, self.alpha_fwd, self.alpha_bwd, self.alpha_bwd)


class DragonMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, use_uscaling: bool):
        super().__init__()
        self.fc_1 = DragonLinear(use_uscaling, hidden_size, intermediate_size, bias=False)
        self.fc_2 = DragonLinear(use_uscaling, intermediate_size, hidden_size, bias=False)
        self.register_buffer("_2_sqrt_5", torch.tensor(2/math.sqrt(5)) if use_uscaling else torch.tensor(1.), persistent=False)

    def forward(self, hidden_states):
        hidden_states = self.fc_1(hidden_states)
        hidden_states = self._2_sqrt_5 * F.relu(hidden_states).square()
        hidden_states = self.fc_2(hidden_states)
        return hidden_states