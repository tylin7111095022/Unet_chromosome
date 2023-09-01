""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *
from .other_network import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ResUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResUnet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResDown(n_channels, 64)
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        self.buttom = Buttom(512, 1024,bilinear)
        self.up1 = ResUp(1024, 512, bilinear)
        self.up2 = ResUp(512, 256, bilinear)
        self.up3 = ResUp(256, 128, bilinear)       
        self.outc = Out(128, n_classes)
        
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.buttom(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x,x1)
        return logits

    
class ConcatModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear:bool=False):
        super(ConcatModel, self).__init__()
        self.unet = UNet(n_channels, n_classes, bilinear)
        self.nest_unet = NestedUNet(n_channels, n_classes,deep_supervision=False)
        self.resunet = ResUnet(n_classes, n_classes, bilinear)

    def forward(self,x):
        middle = self.nest_unet(x)
        out = self.resunet(middle)
        return out

    
if __name__ == '__main__':
    # net = UNet(n_channels=3, n_classes=1)
    # print(net)

    # resunet = ResUnet(n_channels=3, n_classes=1)
    # print(resunet)
    concatmodel = ConcatModel(n_channels=3, n_classes=1, bilinear=False)
    print(concatmodel)
