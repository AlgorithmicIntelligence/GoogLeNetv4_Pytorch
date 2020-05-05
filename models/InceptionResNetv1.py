#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import torch
import torch.nn as nn
from .Layers import Concat_Separable_Conv2d, Separable_Conv2d, Conv2d, Squeeze
from functools import partial

class Inception_ResNet_v1(nn.Module):
    def __init__(self, num_classes):
        super(Inception_ResNet_v1, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            Inceptionv4_stem(),
            
            Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256),
            Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256),
            Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256),
            Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256),
            Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256),
            
            Inception_ResNet_Reduction_A(256, 384, 192, 192, 256),

            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),
            Inception_ResNet_B(896, 128, 128, 128, 128, 896),

            Inception_ResNet_Reduction_B(896, 256, 384, 256, 256, 256, 256),

            Inception_ResNet_C(1792, 192, 192, 192, 192, 1792),
            Inception_ResNet_C(1792, 192, 192, 192, 192, 1792),
            Inception_ResNet_C(1792, 192, 192, 192, 192, 1792),
            Inception_ResNet_C(1792, 192, 192, 192, 192, 1792),
            Inception_ResNet_C(1792, 192, 192, 192, 192, 1792),
            
            nn.AvgPool2d(8, 1),
            nn.Dropout2d(0.2, inplace=True),
            Conv2d(1792, num_classes, kernel_size=1, output=True),
            Squeeze()
        ) 

    def forward(self, x): 
        outputs = self.layers(x)
        return outputs
    
    def init_weights(self, init_mode='VGG'):
        def init_function(m, init_mode):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_mode == 'VGG':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                elif init_mode == 'XAVIER': 
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = (2.0 / float(fan_in + fan_out)) ** 0.5
                    a = (3.0)**0.5 * std
                    with torch.no_grad():
                        m.weight.uniform_(-a, a)
                elif init_mode == 'KAMING':
                     torch.nn.init.kaiming_uniform_(m.weight)                
                m.bias.data.fill_(0)    
        _ = self.apply(partial(init_function, init_mode=init_mode))

class Inceptionv4_stem(nn.Module):
    def __init__(self):
        super(Inceptionv4_stem, self).__init__()
        self.layers = nn.Sequential(Conv2d(3, 32, 3, stride=2),
                                    Conv2d(32, 32, 3, stride=1),
                                    Conv2d(32, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(3, 2),
                                    Conv2d(64, 80, 1),
                                    Conv2d(80, 192, 3),
                                    Conv2d(192, 256, 3, 2)
                                    )

    def forward(self, x):
        x = self.layers(x)
        return x

class Inception_ResNet_A(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel_1, conv3_double_channel_2,
                 conv1_concat_channel):
        super(Inception_ResNet_A, self).__init__()
        
        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)

        self.conv3 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                   Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, padding=1))

        self.conv3_double = nn.Sequential(Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1),
                                          Conv2d(conv3_double_reduce_channel, conv3_double_channel_1, kernel_size=3, padding=1),
                                          Conv2d(conv3_double_channel_1, conv3_double_channel_2, kernel_size=3, padding=1))

        self.conv1_concat = Conv2d(conv1_channel + conv3_channel + conv3_double_channel_2, conv1_concat_channel, 1, output=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv3_double(x)
        x_concat = torch.cat([x1, x2, x3], dim=1)
        outputs = self.conv1_concat(x_concat) + x
        return outputs  

class Inception_ResNet_B(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv7_reduce_channel, conv7_channel_1, conv7_channel_2,
                 conv1_concat_channel):

        super(Inception_ResNet_B, self).__init__()

        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)

        self.conv7 = nn.Sequential(Conv2d(input_channel, conv7_reduce_channel, kernel_size=1),
                                   Separable_Conv2d(conv7_reduce_channel, conv7_channel_1, conv7_channel_2, kernel_size=7, padding=3))

        self.conv1_concat = Conv2d(conv1_channel + conv7_channel_2, conv1_concat_channel, 1, output=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv7(x)
        x_concat = torch.cat([x1, x2], dim=1)
        outputs = self.conv1_concat(x_concat) + x
        return outputs


class Inception_ResNet_C(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel_1, conv3_channel_2, conv1_concat_channel):
        super(Inception_ResNet_C, self).__init__()

        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)

        self.conv3 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                   Separable_Conv2d(conv3_reduce_channel, conv3_channel_1, conv3_channel_2,
                                                    kernel_size=3, padding=1))

        self.conv1_concat = Conv2d(conv1_channel + conv3_channel_2, conv1_concat_channel, 1, output=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x_concat = torch.cat([x1, x2], dim=1)
        outputs = self.conv1_concat(x_concat) + x
        return outputs
  
class Inception_ResNet_Reduction_A(nn.Module):
    def __init__(self, input_channel, conv3_channel, conv3_double_reduce_channel, conv3_double_channel_1, conv3_double_channel_2):
        super(Inception_ResNet_Reduction_A, self).__init__()
        
        self.pool = nn.MaxPool2d(3, 2)
        self.conv3 = Conv2d(input_channel, conv3_channel, 3, 2) # n
        self.conv3_double = nn.Sequential(
            Conv2d(input_channel, conv3_double_reduce_channel, 1), # k
            Conv2d(conv3_double_reduce_channel, conv3_double_channel_1, 3, padding=1), # l
            Conv2d(conv3_double_channel_1, conv3_double_channel_2, 3, stride=2) # m
            )

    def forward(self, x):   
        x1 = self.pool(x)
        x2 = self.conv3(x)
        x3 = self.conv3_double(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Inception_ResNet_Reduction_B(nn.Module):
    def __init__(self, input_channel, conv3_reduce_channel, conv3_channel_1, conv3_channel_2, conv3_double_reduce_channel,
                 conv3_double_channel_1, conv3_double_channel_2):
        super(Inception_ResNet_Reduction_B, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_1 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                     Conv2d(conv3_reduce_channel, conv3_channel_1, kernel_size=3, stride=2))

        self.conv3_2 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                     Conv2d(conv3_reduce_channel, conv3_channel_2, kernel_size=3, stride=2))

        self.conv3_double = nn.Sequential(Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1),
                                          Conv2d(conv3_double_reduce_channel, conv3_double_channel_1, kernel_size=3,
                                                 padding=1),
                                          Conv2d(conv3_double_reduce_channel, conv3_double_channel_2, kernel_size=3,
                                                 stride=2))

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.conv3_1(x)
        x3 = self.conv3_2(x)
        x4 = self.conv3_double(x)
        outputs = torch.cat([x1, x2, x3, x4], dim=1)
        return outputs

if __name__ == '__main__':
    net = GoogLeNetv4(1000).cuda()
    from torchsummary import summary
    summary(net, (3, 299, 299))
