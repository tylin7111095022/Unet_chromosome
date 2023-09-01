import argparse
import logging
import os
import cv2
import numpy as np
import torch
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

#online
from model.other_network import R2U_Net,NestedUNet
#custom
from model.unet_model import UNet, ResUnet, ConcatModel
from dataset import GrayDataset,RGBDataset
from metric import iou,compute_mIoU

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./bestmodel.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--imgpath', '-img',type=str,default=r'dataset\images\A195310_01-01_030719112511_1_1.jpg', help='the path of img')
    parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    return parser.parse_args()


def predict_mask(net,imgpath:str,threshold:float= 0.5):
    net = net.to(device="cpu")
    img = torch.from_numpy(cv2.imread(imgpath)).permute(2,0,1)
    img = img.unsqueeze(0)#加入批次軸
    img = img.to(dtype=torch.float32, device='cpu')
    mask_pred_prob = net(img)
    mask_pred = (torch.sigmoid(mask_pred_prob) > threshold)
    mask_pred = mask_pred.squeeze().numpy()
    mask_pred = mask_pred.astype(np.uint8)*255
    im = Image.fromarray(mask_pred)
    im.save(f"./predict_{os.path.basename(imgpath)}")

    return mask_pred

def evaluate_imgs(net,
                testdataset,
                predict_mask_path,
                out_threshold=0.5,):
    net.eval()
    total_iou = 0
    count = 0
    miou_list = []
    for i,(img, truth) in enumerate(testdataset):
        img = img.unsqueeze(0)#加入批次軸
        img = img.to(dtype=torch.float32)
        truth = truth.unsqueeze(0).to(dtype=torch.int64)#加入批次軸
        #print('shape of truth: ',truth.shape)
        with torch.no_grad():
            mask_pred_prob = net(img)   
            mask_pred = (torch.sigmoid(mask_pred_prob) > out_threshold).to(torch.int64) # (1,1,h ,w)
            #print('shape of mask_pred: ',mask_pred.shape)
            mask = mask_pred.squeeze(0).detach()#(1,h ,w)
            mask *= 255 #把圖片像素轉回255
            #compute the mIOU
        
            miou = compute_mIoU(mask_pred.numpy(), truth.numpy())
            print(miou)
            miou_list.append(miou)
            print('Mean Intersection Over Union: {:6.4f}'.format(miou))

    # return total_iou / count #回傳miou
    return sum(miou_list) / len(miou_list)

if __name__ == '__main__':
    args = get_args()
    # test_img_dir = r'.\dataset\testing\images'
    # test_truth_dir = r'.\dataset\testing\ground_truth'
    # testset = RGBDataset(img_dir = test_img_dir, mask_dir = test_truth_dir)

    net = UNet(n_channels =3,n_classes = 1)
    print(f'Loading model {args.model}')
    net.load_state_dict(torch.load(args.model, map_location="cpu",))
    print('Model loaded!')
    predict_mask(net=net,imgpath=args.imgpath,threshold=0.5)

    # miou = evaluate_imgs(net=net,testdataset=testset,predict_mask_path= args.predictpath,out_threshold=args.mask_threshold)
    # print(f'miou = {miou:5.4f}')
