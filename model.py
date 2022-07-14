#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
model development
featuremap_size = H, W
in_channels = M
out_channels = N
기존 conv 연산량 = H W M K^2 N = HWM(K^2*N) > Deptwise_Sperable_Conv 연산량 = (H W M K^2) + (H W M N) = HWM(K^2+N)
연산량 감소.
'''
import torch
import torch.nn as nn


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super(Depthwise_Separable_Conv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels*kernels_per_layer, 
                                   kernel_size=3, stride=1, padding="same", groups=in_channels, bias=False)
        # padding="same" means that output_shape is equal to be input_shape
        self.pointwise = nn.Conv2d(in_channels*kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(x)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels=None, eps=1e-5, momentum=0.1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2*in_channels if out_channels == None else out_channels


        # res
        self.residual_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2, padding=0, bias=False) 
        # 가우스((in_channel - 1 / 2) + 1)
        self.residual_bnorm = nn.BatchNorm2d(self.out_channels, eps, momentum)

        # Depthwise Sperable Conv
        self.DSC_1 = Depthwise_Separable_Conv(self.in_channels, self.out_channels)
        self.DSC_batch_1 = nn.BatchNorm2d(self.out_channels, eps, momentum)
        self.DSC_activation_1 = nn.ReLU()
        self.DSC_2 = Depthwise_Separable_Conv(self.out_channels, self.out_channels)
        self.DSC_batch_2 = nn.BatchNorm2d(self.out_channels, eps, momentum)
        self.DSC_activation_2 = nn.ReLU()
        self.DSC_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 가우스((in_channel -1 / 2) + 1) 
  
    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bnorm(res)

        x = self.DSC_1(x)
        x = self.DSC_batch_1(x)
        x = self.DSC_activation_1(x)
        x = self.DSC_2(x)
        x = self.DSC_batch_2(x)
        x = self.DSC_maxpool(x)

        return res + x


class mini_Xception(nn.Module):
    def __init__(self, n_class, eps=1e-5, momentum=0.1):
        super(mini_Xception, self).__init__()
        self.name = "mini_xception"
        self.n_class = n_class

        # start with (1,224,224)
        # and no padding
        self.start_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False), # (8, 222, 222)
            nn.BatchNorm2d(8, eps, momentum),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=1, padding=0, bias=False), #(8, 220, 220)
            nn.BatchNorm2d(8, eps, momentum),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([Block(in_channels, None, eps, momentum) for in_channels in [8, 16, 32, 64, 128]])
        # (16, 110, 110) -> (32, 55, 55) -> (64, 28, 28) -> (128, 14, 14) -> (256,7,7)

        self.conv = nn.Conv2d(256, n_class, kernel_size=3, stride=1, padding=1) # (# class, 7, 7)
        self.GAP = nn.AdaptiveAvgPool2d((1,1)) # (1,1) means that target output shpae forces to be (1,1) 
        self.log_softmax = nn.LogSoftmax(dim=1) # // specifically (batch, channel= # class, 1, 1)

    def forward(self, x):
        x = self.start_layers(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv(x)
        x = self.GAP(x)
        x = x.view(-1, x.shape[1]) # (batch, # class, 1, 1) -> (batch, # class)
        x = self.log_softmax(x) # dim=1

        return x
      

