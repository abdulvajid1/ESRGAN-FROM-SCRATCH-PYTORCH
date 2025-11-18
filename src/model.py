import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv Eq
# (input + 2*pad - kernel_size / stride) + 1

class DenseBlock(nn.Module):
    def __init__(self, input_dim, growth_rate, num_layers):
        super(DenseBlock, self).__init__()

    def forward(self, x):
        pass

class RRDB(nn.Module):
    def __init__(self, num_blocks):
        super(RRDB, self).__init__()
        self.dense_layers = nn.Sequential(DenseBlock() for _ in range(num_blocks))

    def forward(self, x):
        x_initial = x
        out = self.dense_layers(x)
        return out + x_initial * 0.2


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

    def forward(self, x):
        pass