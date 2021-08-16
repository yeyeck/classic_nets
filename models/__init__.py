from .alexnet import AlexNet
from .vgg import VGG

__all__ = ['VGG', 'AlexNet']

def create_net(net_cfg, out_features):
    net_type = net_cfg['type']
    if net_type == 'VGG':
        num_layers = net_cfg['num_layers']
        return VGG(num_layers=num_layers, out_features=out_features)
    if net_type == 'AlexNet':
        return AlexNet(out_features)