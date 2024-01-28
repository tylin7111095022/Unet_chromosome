""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

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

####for Res-Unet####
class Dconv(nn.Module):
    def __init__(self,in_c,out_c):
        super(Dconv,self).__init__()
        self.dconv = nn.Sequential(nn.Conv2d(in_c, out_c,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(out_c),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1),
                                    nn.BatchNorm2d(out_c),
                                    nn.ReLU(inplace=True))
    def forward(self,x):
        return self.dconv(x)


class ResDown(nn.Module):
    def __init__(self,in_c,out_c):
        super(ResDown, self).__init__()
        self.doubleconv = Dconv(in_c,out_c)
        self.skipconv = nn.Conv2d(in_c,out_c,kernel_size=(1,1),padding=0)

    def forward(self,x):
        x1 = F.max_pool2d(self.doubleconv(x),2)
        skip = F.max_pool2d(self.skipconv(x),2)
        out = x1 + skip
        return out
    
class Buttom(nn.Module):
    def __init__(self,in_c,out_c,bilinear=True):
        super(Buttom,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(out_c, out_c//2, kernel_size=(2,2), stride=2)
        self.dconv = Dconv(in_c,out_c)
        self.skipconv = nn.ConvTranspose2d(in_c, out_c//2, kernel_size=(2,2), stride=2)

    def forward(self, x):
        x1 = self.dconv(x)
        x2 = self.up(x1)
        skip = self.skipconv(x)
        return skip + x2

class ResUp(nn.Module):
    """concat x1 and x2 then doubleconvolution and transconvolution"""

    def __init__(self, in_c, out_c, bilinear=True):
        super(ResUp,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(out_c, out_c // 2, kernel_size=2, stride=2)

        self.dconv = Dconv(in_c,out_c)
        self.upconv = nn.ConvTranspose2d(in_c//2, in_c//2,kernel_size=(2,2),stride=2)
        self.skiptransconv = nn.ConvTranspose2d(in_c // 2, out_c // 2, kernel_size=(2,2),stride=2)

    def forward(self, x1, x2):
        x2 = self.upconv(x2)
        x = torch.cat([x1,x2],dim=1)
        x = self.dconv(x)
        x = self.up(x)
        skip = self.skiptransconv(x1)
        return x + skip

class Out(nn.Module):
    def __init__(self,in_c, out_map):
        super(Out,self).__init__()

        self.dconv = Dconv(in_c,in_c//2)
        self.conv2 = nn.Conv2d(in_c//2, in_c//2, kernel_size=3, padding=1)
        self.outconv = nn.Conv2d(in_c//2, out_map, kernel_size=1,)
        self.upconv = nn.ConvTranspose2d(in_c//2, in_c//2,kernel_size=(2,2),stride=2)


    def forward(self,x1,x2):
        x2 = self.upconv(x2)
        x = torch.cat([x1,x2],dim=1)
        x = self.dconv(x)
        x = self.conv2(x)
        out = self.outconv(x)

        return out
