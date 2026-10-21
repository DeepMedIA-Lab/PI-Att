import torch
import torch.nn as nn
import numpy as np
import copy

"""
This Code includes:
Network Construction Codes
"""


#Block for convolution
def conv_block(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True)
    )

def conv_block_with_batchnorm(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_block_with_dropout(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        #nn.Dropout(0.2),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),        
        nn.ReLU(inplace=True),

        nn.Dropout(0.2),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),        
        nn.ReLU(inplace=True)
    )


class up_process(nn.Module):
    def __init__(self, selected_block, reverse_feat:list, kernel_pool=2, stride_pool=2):
        super().__init__()

        self.upsample = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(reverse_feat[i], reverse_feat[i + 1], kernel_size = kernel_pool, stride = stride_pool),
                                            selected_block(reverse_feat[i], reverse_feat[i+1])) for i in range(4)])


    def forward(self, x, cat_values:list):
        for index in range(4):
            x = self.upsample[index][0](x)
            x = torch.cat([x, cat_values[-(index+1)]], dim=1)
            x = self.upsample[index][1](x)

        return(x)


class UNet(nn.Module):
    def __init__(self, given_feature_size:list, parameters, block_type_name = "default", kernel_pool=2, stride_pool=2):    
        super().__init__()
        #Block Type Selection
        block_type = {"default":conv_block, "dropout":conv_block_with_dropout, "batchnorm":conv_block_with_batchnorm}
        selected_block = block_type[block_type_name]

        #Model Parameters
        self.feature_size = given_feature_size
        self.reverse_feat = self.feature_size[::-1]
        self.layer_size = len(self.feature_size) # +1 for last bottle convolution layer
        self.hyper_parameters = parameters
        self.total_time = 0
        self.prediction_results = []
        self.training_performance = []
        self.used_ph_version = None

        #Model modules
        #Down-direction
        self.down_process = nn.ModuleList([nn.Sequential(selected_block(self.feature_size[i], self.feature_size[i+1]),
                                          nn.MaxPool2d(kernel_size = kernel_pool, stride = stride_pool)) for i in range(4)])

        #Bottle-direction
        self.bottleneck_convolution = selected_block(self.feature_size[-2], self.feature_size[-1])

        #Up-direction(Multi_task)
        self.up_process1 = up_process(selected_block, self.reverse_feat, kernel_pool, stride_pool)
        self.last_conv = nn.Conv2d(self.feature_size[1], 1, kernel_size=1)

    def forward(self, image):
        cat_values = []
        #Down-Process
        for index in range(4):
            image = self.down_process[index][0](image)
            cat_values.append(image)
            image = self.down_process[index][1](image)
        
        #Bottle-neck Process
        bottleneck_image = self.bottleneck_convolution(image)

        #Up-Process
        image = self.up_process1(bottleneck_image, cat_values)

        #Prediction(Multi_task)
        segmentation_image = torch.sigmoid(self.last_conv(image))

        return segmentation_image