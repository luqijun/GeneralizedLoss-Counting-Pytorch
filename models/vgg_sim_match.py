import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torchvision import models
from torch.nn import functional as F
from .layers import *

seg_layer_name = 'SegmentationLayer'
gen_kernel_layer_name = 'GenerateKernelLayer21'

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG_Sim_Match(nn.Module):
    def __init__(self, features, down=8, o_cn=1, final='abs', **kwargs):
        super(VGG_Sim_Match, self).__init__()
        self.down = down
        self.final = final
        
        # self.reg_layer = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, o_cn, 1)
        # )
        
        self.trans_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.sim_conv_layer = eval(gen_kernel_layer_name)([512], kernel_size=3, hidden_channels=256)
        self.seg_layer = eval(seg_layer_name)(in_channels=256, out_channels=1, hidden_channels=128)
        
        self._initialize_weights()
        self.features = features



    def forward(self, x, **kwargs):
        if x.shape[-1]!=512 or x.shape[-2]!=512:
            pass
        x = self.features(x)
        if self.down < 16:
            x = F.interpolate(x, scale_factor=2)

        x, seg = self.sim_conv(x)

        # x = self.reg_layer(x)
        if self.final == 'abs':
            x = torch.abs(x)
        elif self.final == 'relu':
            x = torch.relu(x)

        outputs = {}
        outputs['pre_den'] = x
        outputs['pre_seg'] = seg

        return outputs
    
    
    def sim_conv(self, x):
        x = self.trans_layer(x)
        seg = self.seg_layer(x)
        x = x * seg
        
        kernel = self.sim_conv_layer(x)
        x = self.sim_conv_layer.conv_with_kernels(x, 0)
        
        return x, seg
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


