from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        a = x
        x = self.block(x)
        return a + x

class ResidualBottleneck(nn.Module):
    def __init__(self, channles:int):
        super(ResidualBottleneck, self).__init__()
        bottle_channels = channles / 4
        bottle_channels = max(bottle_channels, 64)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=channles, out_channels=bottle_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bottle_channels, out_channels=bottle_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bottle_channels, out_channels=channles, kernel_size=1, stride=1)
        )

    def forward(self, x):
        a = x
        x = self.bottleneck(x)
        return a + x


if __name__ == '__main__':
    input = torch.randn(3, 128, 443, 443)
    block = ResidualBlock(128)
    print(block(input).size())

    bottleneck = ResidualBottleneck(128)
    print(bottleneck)
    print(bottleneck(input).size())