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
    
def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, 1, h, w) or (n, h, w)
    """
    target = target.long()
    if target.dim() == 4:
        target = target[:,0,:,:]
    assert not target.requires_grad
    assert predict.dim() == 4
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    # print(f" target_mask shape: {target_mask.shape}") #(B,H,W)
    # print(target_mask)
    target = target[target_mask]
    # print(f" label shape: {target.shape}")
    if not target.data.dim():
        return torch.zeros(1)
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous() # (n,c,h,w) -> (n,h,w,c)
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

    
if __name__ == '__main__':
    # net = UNet(n_channels=3, n_classes=1)
    # print(net)

    # resunet = ResUnet(n_channels=3, n_classes=1)
    # print(resunet)
    concatmodel = ConcatModel(n_channels=3, n_classes=1, bilinear=False)
    print(concatmodel)
