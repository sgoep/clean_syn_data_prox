# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from radon_operator import filter_sinogram
import torch_radon

from .custom_layers import null_space_layer, proximal_layer, range_layer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, constraint, null_space, norm2=None, bilinear=True):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.constraint = constraint
            
        self.null_space = null_space

        N = 128
        Nal = 80
        angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
        radon = torch_radon.Radon(128, angles, det_count=128, det_spacing=1, clip_to_circle=True)
        self.A = lambda x: radon.forward(x)
        self.FBP = lambda y: radon.backward(filter_sinogram(y))
        
        if self.null_space:
            self.null_space_layer = null_space_layer()
            if self.constraint:
                # self.ell2_norm = torch.Tensor(np.load('data_htc2022_simulated/norm2.npy', allow_pickle=True)).to(config.device)
                self.ell2_norm = torch.Tensor(norm2).to(config.device)
                self.proximal_layer = proximal_layer(self.ell2_norm)      

    def forward(self, X):
        x0  = X
        
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        x_res = torch.clone(x)
    
        if self.null_space:
            # print("nsn")
            x_nsn = self.null_space_layer(x_res, self.A, self.FBP)
            xout = x_nsn
        if self.constraint:
            # print("dpnsn")
            x_dp = self.proximal_layer(x_res, self.A, self.FBP)
            xout = x_nsn + x_dp
        if (not self.null_space) and (not self.constraint):
            # print("resnet")
            xout = x_res
        
        # x_res = torch.clone(x)
        # xout = x_res
        # if self.null_space:
        #     x_nsn = self.null_space_layer(x_res, self.A, self.FBP)
        #     xout = x_nsn
        # if self.constraint:
        #     x_dp = self.proximal_layer(x_res, self.A, self.FBP)
        #     xout = x_nsn + x_dp
            
        return x0 + xout, xout

