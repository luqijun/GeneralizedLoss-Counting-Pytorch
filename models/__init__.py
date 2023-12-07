
from .vgg import VGG
from .vgg_sim_match import VGG_Sim_Match
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],    
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],    
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512]
}

def vgg19(model_name='VGG', down=8, bn=False, o_cn=1, final='abs'):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = eval(model_name)(make_layers(cfg['E'], batch_norm=False), down=down, o_cn=o_cn, final=final)
    
    state_dict = model_zoo.load_url(model_urls['vgg19'])
    model.load_state_dict(state_dict, strict=False)
    return model