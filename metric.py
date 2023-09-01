import torch
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
'''
dice_score = (2*precision*recall)/ (precision + recall) 
dice_score越接近1越好

https://chih-sheng-huang821.medium.com/%E5%BD%B1%E5%83%8F%E5%88%87%E5%89%B2%E4%BB%BB%E5%8B%99%E5%B8%B8%E7%94%A8%E7%9A%84%E6%8C%87%E6%A8%99-iou%E5%92%8Cdice-coefficient-3fcc1a89cd1c
'''

SMOOTH = sys.float_info.min

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x H x W shape
    assert outputs.dim() == labels.dim()
    if outputs.dim() == 4:
        total_intersection = 0
        total_union = 0
        for i in range(outputs.size(0)):
            inter = torch.sum((outputs[i] & labels[i]).float(),dim=(0,1,2))
            total_intersection += inter
            #print(f'inter:{inter}')
            union = torch.sum((outputs[i] | labels[i]).float(),dim=(0, 1, 2))
            #print(f'union:{union}')
            total_union += union
        iou_score = (total_intersection + SMOOTH) / (total_union + SMOOTH)  # We smooth our devision to avoid 0/0

    else: #outputs.dim() == labels.dim() == 3 #沒有batch軸

        intersection = torch.sum((outputs & labels).float(),dim=(1,2))  # Will be zero if Truth=0 or Prediction=0
        #print(f'intersection: {intersection}')

        union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
        #print(f'union: {union}')
        
        iou_score = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    return iou_score  # Or thresholded.mean() if you are interested in average across the batch

def compute_mIoU(pred, label):

    cf_m = confusion_matrix(label.flatten(), pred.flatten())

    #print(cf_m)
    intersection = np.diag(cf_m)  # TP + FN
    union = np.sum(cf_m, axis=1) + np.sum(cf_m, axis=0) - intersection
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    return mIoU
