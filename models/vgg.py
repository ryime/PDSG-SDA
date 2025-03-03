import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from timm.models import register_model

class VGG11(nn.Module):
    def __init__(self, num_classes=10, img_size=(3,32,32), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=img_size[0], out_channels=64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=64),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=256),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Flatten(),
            layer.Linear(in_features=4*4*512, out_features=4096),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Linear(in_features=4096, out_features=4096),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Linear(in_features=4096, out_features=num_classes),
            )
        functional.set_step_mode(self.net, 'm')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.net(x)

class VGGSNN(nn.Module):
    def __init__(self, num_classes=10, img_size=(3, 32, 32), **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=img_size[0], out_channels=64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=64),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=256),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=256),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=512),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),
            layer.AdaptiveAvgPool2d((1,1)),
            layer.Flatten(),
            layer.Linear(in_features=512, out_features=num_classes))
        
        functional.set_step_mode(self.net, 'm')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.net(x)


@register_model    
def spiking_vgg11(**kwargs):
    return VGG11(**kwargs)

@register_model
def spiking_vggsnn(**kwargs):
    return VGGSNN(**kwargs)

class heaviside_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError('Heaviside does not contain backward function')
    
class Heaviside(nn.Module):
    def forward(self, x: torch.Tensor):
        return heaviside_function(x)