import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class GrayDataset(Dataset):
    f'''
    回傳img(torch.uint8)及label(torch.float32)\n
    這個dataset 最後回傳的圖片為灰階圖
    '''
    def __init__(self,img_dir,mask_dir,transform = None):
        assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir)), "numbers of img and label dismatch."
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = [i for i in os.listdir(img_dir)]
        
        self.transforms = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_a = cv2.imread(os.path.join(self.img_dir, img_name),cv2.IMREAD_GRAYSCALE)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #如果沒通道軸，加入通道軸
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))
        img = img.to(torch.float32) / 255 #[0, 255] -> [0, 1]
        imgsize = (img.shape[1], img.shape[2])
        label_name = img_name
        label_path = os.path.join(self.mask_dir, label_name)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        if label.ndim == 2:
            label = np.expand_dims(label,2) #如果沒通道軸，加入通道軸
        label = cv2.resize(label, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將label resize成跟 img  一樣的長寬
        #label內的值不只兩個，這導致除以255後值介於0~1的值在後續計算iou將label轉回int64的時候某些值被無條件捨去成0
        ret,label_binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
        #print(np.unique(label_binary))
        label_t = torch.from_numpy(label_binary).unsqueeze(0).to(torch.float32)#加入通道軸
        # 處理標籤，将像素值255改為1
        if label_t.max() > 1:
            label_t[label_t == 255] = 1

        if self.transforms:
            img = self.transform(img)
            label_t = self.transform(label_t)

        return img,label_t

    def __len__(self):
        return len(self.img_list)



if __name__ == "__main__":
    ds = GrayDataset(img_dir = './dataset/images', mask_dir= './dataset/masks')
    img, mask = ds[10]
    print(img.shape)
    print(mask.shape)

    cv2.imwrite("img.jpg", img.permute(1,2,0).numpy())
    cv2.imwrite("mask.jpg", (mask*255).permute(1,2,0).numpy(),)