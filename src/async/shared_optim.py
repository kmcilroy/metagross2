import math
import torch
from torch import optim


class SharedAdam(optim.Adam):
    """Adam with optimizer state in shared memory (for multiprocessing Hogwild)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Move optimizer state tensors to shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state.setdefault('step', torch.zeros(1))
                state.setdefault('exp_avg', torch.zeros_like(p.data))
                state.setdefault('exp_avg_sq', torch.zeros_like(p.data))
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()