import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv Eq
# (input + 2*pad - kernel_size / stride) + 1
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True)
            )
    def forward(self, x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvBlock(64, 64, 3, 1, 1)
        self.conv2 = ConvBlock(128, 64, 3, 1, 1)
        self.conv3 = ConvBlock(192, 64, 3, 1, 1)

    def forward(self, x):
        x_initial = x
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat([x, out1], 1))
        out3 = self.conv3(torch.cat([x, out1, out2], 1))
        return out3 + x_initial * 0.2
    
class RRDB(nn.Module):
    def __init__(self, num_blocks):
        super(RRDB, self).__init__()
        self.dense_layers = nn.Sequential(
            *[DenseBlock() for _ in range(num_blocks)]
            )

    def forward(self, x):
        x_initial = x
        out = self.dense_layers(x)
        return out + x_initial * 0.2

class UpSampleBlock(nn.Module):
    def __init__(self):
        super(UpSampleBlock, self).__init__()
        self.final_block = nn.Sequential(
            nn.Conv2d(64, 64 * 16, 3, 1, 1),
            nn.PixelShuffle(4),
            ConvBlock(64, 64, 3, 1, 1),
            ConvBlock(64, 3, 3, 1, 1),
            ConvBlock(3, 3, 1, 1, 0)
            )

    def forward(self, x):
        return self.final_block(x)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.body = nn.Sequential(
            *[RRDB(num_blocks=3) for _ in range(5)],
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
        
        self.head = UpSampleBlock()

    def forward(self, x):
        out1 = self.initial(x)
        body_out = self.body(out1) + out1
        out = self.head(body_out)
        return F.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_blocks=3):
        super(Discriminator, self).__init__()
        self.initial = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)

        self.dense_layers = nn.Sequential(
            *[DenseBlock() for _ in range(num_blocks)]
            )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        out = self.initial(x)
        out = self.dense_layers(out)
        out = self.final(out)
        return out



def main():
    generator = Generator(input_dim=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)
    x = torch.randn((1, 3, 128, 128)).to(device)
    out = generator(x)
    print("Input Shape", x.shape)
    print("Out shape", out.shape)

    discriminator = Discriminator(input_dim=3)
    discriminator.to(device)
    disc_out = discriminator(out)
    print("Discriminator Out shape", disc_out.shape)


if __name__ == "__main__":
    main()