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

class GoogLeNetv4(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(GoogLeNetv4, self).__init__()
        self.num_classes = num_classes
        self.mode = mode     
        self.layers = nn.Sequential(
            Inceptionv4_stem(),
            
            Inceptionv4_A(384, 96, 96, 64, 96, 64, 96, 96),
            Inceptionv4_A(384, 96, 96, 64, 96, 64, 96, 96),
            Inceptionv4_A(384, 96, 96, 64, 96, 64, 96, 96),
            Inceptionv4_A(384, 96, 96, 64, 96, 64, 96, 96),
            
            Inceptionv4_reduction_A(384, 384, 192, 224, 256),
            
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
            Inceptionv4_B(1024, 128, 384, 192, 224, 256, 192, 224, 256),
        
            Inceptionv4_reduction_B(1024, 192, 256, 320),    
            
            Inceptionv4_C(1536, 256, 256, 384, 256, 384, 448, 512, 256),
            Inceptionv4_C(1536, 256, 256, 384, 256, 384, 448, 512, 256),
            Inceptionv4_C(1536, 256, 256, 384, 256, 384, 448, 512, 256),
            
            nn.AvgPool2d(8, 1),
            nn.Dropout2d(0.2, inplace=True),
            Conv2d(1536, num_classes, kernel_size=1, output=True),
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
        self.stem1 = nn.Sequential(
            Conv2d(3, 32, 3, stride=2),
            Conv2d(32, 32, 3, stride=1),
            Conv2d(32, 64, 3, stride=1, padding=1),
            )
        self.stem1_1 = nn.MaxPool2d(3, 2)
        self.stem1_2 = Conv2d(64, 96, 3, 2)
        self.stem2_1 = nn.Sequential(
            Conv2d(160, 64, 1),
            Conv2d(64, 96, 3),
            )
        self.stem2_2 = nn.Sequential(
            Conv2d(160, 64, 1),
            Separable_Conv2d(64, 64, 64, 7, padding=3),
            Conv2d(64, 96, 3)
            )
        self.stem3_1 = nn.MaxPool2d(3, 2)
        self.stem3_2 = Conv2d(192, 192 ,3, 2)

    def forward(self, x):   
        x = self.stem1(x)
        x1 = self.stem1_1(x)
        x2 = self.stem1_2(x)
        x = torch.cat([x1 , x2], dim=1)
        x1 = self.stem2_1(x)
        x2 = self.stem2_2(x)
        x = torch.cat([x1, x2], dim=1)
        x1 = self.stem3_1(x)
        x2 = self.stem3_2(x)
        x = torch.cat([x1, x2], dim=1)
        return x

class Inceptionv4_A(nn.Module):
    def __init__(self, input_channel, pool_reduce_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel_1, conv3_double_channel_2):

        super(Inceptionv4_A, self).__init__()

        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  Conv2d(input_channel, pool_reduce_channel, kernel_size=1))
        
        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        
        self.conv3 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                   Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, padding=1))

        self.conv3_double = nn.Sequential(Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1),
                                          Conv2d(conv3_double_reduce_channel, conv3_double_channel_1, kernel_size=3, padding=1),
                                          Conv2d(conv3_double_channel_1, conv3_double_channel_2, kernel_size=3, padding=1))
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv3_double(x)
        x4 = self.pool(x)
        outputs = torch.cat([x1, x2, x3, x4], dim=1)
        return outputs  

class Inceptionv4_B(nn.Module):
    def __init__(self, input_channel, pool_reduce_channel, conv1_channel, conv7_reduce_channel,
                 conv7_channel_1, conv7_channel_2, conv7_double_reduce_channel, conv7_double_channel_1, conv7_double_channel_2):

        super(Inceptionv4_B, self).__init__()

        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        
        self.conv7 = nn.Sequential(Conv2d(input_channel, conv7_reduce_channel, kernel_size=1),
                                   Separable_Conv2d(conv7_reduce_channel, conv7_channel_1, conv7_channel_2, kernel_size=7, padding=3))

        self.conv7_double = nn.Sequential(Conv2d(input_channel, conv7_double_reduce_channel, kernel_size=1),
                                          Separable_Conv2d(conv7_double_reduce_channel, conv7_double_reduce_channel, conv7_double_channel_1, kernel_size=7, padding=3),
                                          Separable_Conv2d(conv7_double_channel_1, conv7_double_channel_1, conv7_double_channel_2, kernel_size=7, padding=3))
        
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  Conv2d(input_channel, pool_reduce_channel, kernel_size=1))
    
    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv7(x)
        x3 = self.conv7_double(x)
        x4 = self.pool(x)
        outputs = torch.cat([x1, x2, x3, x4], dim=1)
        return outputs  
  
class Inceptionv4_C(nn.Module):
    def __init__(self, input_channel, pool_reduce_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel_1_1, conv3_double_channel_1_2, conv3_double_channel_2):

        super(Inceptionv4_C, self).__init__()

        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  Conv2d(input_channel, pool_reduce_channel, kernel_size=1))
        
        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        
        self.conv3 = nn.Sequential(Conv2d(input_channel, conv3_reduce_channel, kernel_size=1),
                                   Concat_Separable_Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, padding=1))

        self.conv3_double = nn.Sequential(Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1),
                                          Separable_Conv2d(conv3_double_reduce_channel, conv3_double_channel_1_1, conv3_double_channel_1_2, kernel_size=3, padding=1),
                                          Concat_Separable_Conv2d(conv3_double_channel_1_2, conv3_double_channel_2, kernel_size=3, padding=1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv3_double(x)
        x4 = self.pool(x)
        outputs = torch.cat([x1, x2, x3, x4], dim=1)
        return outputs      
  
class Inceptionv4_reduction_A(nn.Module):
    def __init__(self, input_channel, conv3_channel, conv3_double_reduce_channel, conv3_double_channel_1, conv3_double_channel_2):
        super(Inceptionv4_reduction_A, self).__init__()
        
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
  
class Inceptionv4_reduction_B(nn.Module):
    def __init__(self, input_channel, conv3_channel, conv3_double_reduce_channel, conv3_double_channel):
        super(Inceptionv4_reduction_B, self).__init__()
        
        self.pool = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Sequential(Conv2d(input_channel, conv3_channel, 1),
                                   Conv2d(conv3_channel, conv3_channel, 3, 2))
        self.conv3_double = nn.Sequential(Conv2d(input_channel, conv3_double_reduce_channel, 1),
                                          Separable_Conv2d(conv3_double_reduce_channel, conv3_double_reduce_channel, conv3_double_channel, 7, padding=3),
                                          Conv2d(conv3_double_channel, conv3_double_channel, 3, stride=2))

    def forward(self, x):   
        
        x1 = self.pool(x)
        x2 = self.conv3(x)
        x3 = self.conv3_double(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x    

if __name__ == '__main__':
    net = GoogLeNetv4(1000).cuda()
    from torchsummary import summary
    summary(net, (3, 299, 299))
