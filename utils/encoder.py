import torch
import torch.nn as nn

def get_encoder(encoder_name: str, timestep: int=10):
    if encoder_name == 'direct':
        return DirectEncoder(timestep)
    elif encoder_name == 'poisson':
        return PoissonEncoder(timestep)
    elif encoder_name == 'integer':
        return DVSEncoder()
    elif encoder_name == 'binary':
        return DVSSignEncoder()
    else:
        raise NotImplementedError(encoder_name)
        
class Poisson_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.rand_like(x).le(x).to(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output
    
class Sign_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return (x >= 1.0).to(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output

class DirectEncoder(nn.Module):
    def __init__(self, timestep: int=10):
        super().__init__()
        self.timestep = timestep
    
    def forward(self, x: torch.Tensor):
        return torch.repeat_interleave(x.unsqueeze(0), self.timestep, dim=0)
    
class PoissonEncoder(nn.Module):
    def __init__(self, timestep: int=10):
        super().__init__()
        self.timestep = timestep
    
    def forward(self, x: torch.Tensor):
        x = (x - x.min()) / (x.max() - x.min())
        y = torch.repeat_interleave(x.unsqueeze(0), self.timestep, dim=0)
        y = Poisson_STE.apply(y)
        return y
    
class DVSSignEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        y = x.transpose(0, 1)
        y = Sign_STE.apply(y)
        return y
    
class DVSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        y = x.transpose(0, 1)
        return y
