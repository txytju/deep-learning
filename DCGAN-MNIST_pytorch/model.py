import torch as t
from torch import nn

# from torchvision.datasets import MNIST
# from torchvision import transforms as T
# from torch.utils.data import DataLoader
from torch.autograd import Variable as V

import numpy as np

class NetG(nn.Module):
    """
    生成器定义
    """

    def __init__(self):
        super(NetG, self).__init__()
        ngf = 64  # 生成器feature map数
        nz = 100
        
        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(in_channels=nz, 
                               out_channels=ngf*4, 
                               kernel_size=2, 
                               stride=1, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 2 x 2

            nn.ConvTranspose2d(in_channels=ngf*4, 
                               out_channels=ngf*2,
                               kernel_size=4, 
                               stride=3, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 7 x 7

            nn.ConvTranspose2d(in_channels=ngf*2,
                               out_channels=ngf,
                               kernel_size=4,
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 14 x 14

            nn.ConvTranspose2d(in_channels=ngf,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            # 上一步的输出形状：1 x 28 x 28
        )

    def forward(self, input):
        return self.main(input)

class NetD(nn.Module):

    def __init__(self):
        super(NetD, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # 输入 1 x 28 x 28 
            nn.Conv2d(1, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 输出 (ndf) x 14 x 14

            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 输出 (ndf*2) x 7 x 7

            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 输出 (ndf*4) x 3 x 3

            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )
        
        self.net_1 = nn.Sequential(
            # 输入 3 x 28 x 28 
            # 输出 (ndf) x 14 x 14
            nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2))

        self.net_2 = nn.Sequential(
            # 输出 (ndf*2) x 7 x 7
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2))

        self.net_3 = nn.Sequential(
            # 输出 (ndf*4) x 3 x 3
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2))

    def forward(self, input):
         return self.main(input).view(-1)