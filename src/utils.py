import torch.nn as nn
import torch
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = vgg19(pretrained=True).features[:34] # before relu
        for params in self.vgg19.parameters():
            params.required_grad = False
        
        self.mse = nn.MSELoss()
    
    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real = self.vgg19(real)
        fake = self.vgg19(fake)
        return self.mse(real, fake)
        