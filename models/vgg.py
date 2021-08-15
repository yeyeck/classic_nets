from torch import nn
import torch


def conv_3x3s1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

def point_conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


class VGG(nn.Module):
    def __init__(self, num_layers=16, out_features=256, use_point_conv=False):
        super(VGG, self).__init__()
        if num_layers == 16:
            self.layers = (2, 2, 3, 3, 3)
        elif num_layers == 19:
            self.layers = (2, 2, 4, 4, 4)
        elif num_layers == 11:
            self.layers = (1, 1, 2, 2, 2)
        elif num_layers == 13:
            self.layers = (2, 2, 2, 2, 2)
        
        layers = []
        in_channels = 3
        out_channels = 64
        for i in self.layers:
            for _ in range(i):
                layers.append(conv_3x3s1(in_channels, out_channels))
                layers.append(nn.ReLU(inplace=True))
                if use_point_conv:
                    layers.append(point_conv(in_channels=out_channels, out_channels=out_channels))
                    layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            out_channels = out_channels * 2
            out_channels = min(out_channels, 512)
            
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 49, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_features)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    from torchvision import models
    model_ = models.vgg16()
    print(model_)
    inputs = torch.randn(3, 3, 224, 224)
    model = VGG(16)
    print(model)
    print(model(inputs).size())


                

