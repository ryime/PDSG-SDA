import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from timm.models import register_model

@register_model    
def spiking_resnet18(**kwargs):
    return ResNet18(**kwargs)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, img_size=(3,32,32), **kwargs):
        super(ResNet18, self).__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=img_size[0], out_channels=64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()), # unknown surrogate function
            ResidualBlock(in_channels=64, out_channels=64, block_number=2),
            ResidualBlock(in_channels=64, out_channels=128, block_number=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=256, block_number=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=512, block_number=2, stride=2),
            layer.AdaptiveAvgPool2d((1,1)),
            layer.Flatten(),
            layer.Linear(in_features=512, out_features=num_classes)
        )
        functional.set_step_mode(self, 'm')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.net(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_path = nn.Sequential(
            layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside()),  # unknown surrogate function
            layer.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut_path = nn.Sequential(
                layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut_path = nn.Identity()
        
        self.output_neuron = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=Heaviside())  # unknown surrogate function

    def forward(self, x: torch.Tensor):
        return self.output_neuron(self.residual_path(x) + self.shortcut_path(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_number, stride=1):
        super(ResidualBlock, self).__init__()
        
        layers = []
        layers.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
        
        for i in range(1, block_number):
            layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.block(x)
    
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

