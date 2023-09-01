import argparse
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader,random_split #random_split幫助切割dataset
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings("ignore")

#online
from model.other_network import R2U_Net,NestedUNet
#custom module
from metric import iou, compute_mIoU
from model.unet_model import UNet
from dataset import GrayDataset,RGBDataset

dir_img = '.\\dataset\\zong\\train2017' #訓練集的圖片所在路徑
dir_truth = '.\\dataset\\zong\\train_mask' #訓練集的真實label所在路徑
dir_checkpoint = '.\\' #儲存模型的權重檔所在路徑
dir_predict = './dataset/validation_predict' #驗證過程中儲存模型預測的predict mask

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=3,dest='in_channel',help="channels of input images")
    parser.add_argument('--epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--batch','-b',type=int,metavar='B',dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=1,help='Number of classes')
    parser.add_argument('--load', '-l', action='store_true', default=False, help='Load model from a .pth file')
    parser.add_argument('--rate_of_learning','-r',type = float, dest='lr', default=1e-2,help='learning rate of model')
    parser.add_argument('--no_save_pic','-n',action='store_false',default=True,dest='no_save',help='don\'t save the validation picture during the training.')
    parser.add_argument('--log_name', type=str,default='./log.txt',help='filename of log')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset = RGBDataset(img_dir = dir_img, mask_dir= dir_truth)

    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(args.log_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=args.in_channel,n_classes = args.classes)
    logging.info(model)
    
    if args.load:
        load_path = 'mymodel.pth'
        model.load_state_dict(torch.load(load_path, map_location=device))
        logging.info(f'Model loaded from {load_path}')

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas=(0.9,0.999))
    criterion = torch.nn.BCEWithLogitsLoss()
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    Parameters of training:
        model: Unet
        in_channel: {args.in_channel}
        output_map: {args.classes}
        type of image: RGB
        learning rate: {args.lr}
        optimizer : Adam
        loss function : BCEWithLogitsLoss
    =======================================
    ''')
    try:
        training(net=model,
                loss_fn = criterion,
                optimizer = optimizer,
                dataset = trainingDataset,
                epoch=args.epoch,
                batch_size=args.batch_size,
                device=device,
                save_checkpoint= True,
                save_picture=args.no_save,
                is_valid = True)
                
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    return

def training(net,
            loss_fn,
            optimizer,
            dataset,
            epoch:int,
            batch_size:int,
            device,
            save_checkpoint: bool = True,
            save_picture : bool = True,
            is_valid: bool = True):

    arg_loader = dict(batch_size = batch_size, num_workers = 0)
    if is_valid:
        n_train = int(0.8*len(dataset))
        n_valid = len(dataset) - n_train
        trainset, validset = random_split(dataset,[n_train, n_valid],generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(trainset,shuffle = True, **arg_loader)
        valid_loader = DataLoader(validset,shuffle = False, **arg_loader)
    else:
        n_train = len(dataset)
        n_valid = 0
        train_loader = DataLoader(dataset,shuffle = True, **arg_loader)
    #Initial logging
    logging.info(f'''Starting training:
        Epochs:          {epoch}
        Batch size:      {batch_size}
        Training size:   {n_train}
        validation size: {n_valid}
        checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,verbose=True) # goal: maximize evaluation criteria
    net.to(device)
    #begin to train model
    currentIou = sys.float_info.min
    for i in range(1, epoch+1):
        net.train()
        epoch_loss = 0
        train_iou = 0
        for imgs, truthes in tqdm(train_loader):
            imgs = imgs.to(torch.float32)
            imgs = imgs.to(device)
            truthes = truthes.to(device = device)
            predict_mask = net(imgs) #Size([B, 1, 512, 512]

            train_result = (torch.sigmoid(predict_mask) > 0.5).to(torch.int64)
            train_iou += iou(train_result,truthes.to(torch.int64))

            loss = loss_fn(predict_mask, truthes)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f'Training loss: {epoch_loss:6.4f} at epoch {i}.')

        #validation
        if is_valid:
            net.eval()           
            miou_list = []
            for val_imgs, val_truthes in valid_loader:
                val_imgs = val_imgs.to(dtype =torch.float32,device=device)
                val_truthes = val_truthes.to(torch.int64).to(device)
                with torch.no_grad():
                    logits = net(val_imgs)
                    #print(mask_pred.cpu())
                    mask_pred = (torch.sigmoid(logits) > 0.5).to(torch.int64) # (batch, channel,h ,w)
                    if save_picture:
                        if i % 10 == 0:
                            #看一下mask_pred的圖
                            # cv2.imshow('"mask_pred"',mask_pred[0].permute(1,2,0).cpu().numpy()) #取驗證集的第一張圖片預測之mask看效果
                            # cv2.waitKey(0) # wait for ay key to exit window
                            mask = mask_pred.cpu().detach()
                            mask *= 255 #把圖片像素轉回255
                            cv2.imwrite(os.path.join(dir_predict,f'epoch{i}.png'),mask[0].permute(1,2,0).cpu().numpy())
                            #logging.info(f"epoch{i}.png had written.")
                    #compute the IOU           
            #         miou_score = iou(mask_pred,val_truthes)
                    miou_score = compute_mIoU(mask_pred.cpu().numpy(),val_truthes.cpu().numpy()) #新增
                    miou_list.append(miou_score)
        
            valid_miou = sum(miou_list)/len(miou_list)
            scheduler.step(valid_miou) #根據驗證資料集的miou有沒有上升決定

            logging.info('Mean Intersection Over Union of Validation: {:6.4f} %'.format(valid_miou*100))
            if (save_checkpoint) and (valid_miou > currentIou):
                currentIou = valid_miou
                torch.save(net.state_dict(), os.path.join(dir_checkpoint,'bestmodel.pth'.format(i)))
                logging.info(f'Model saved at epoch {i}.')
    logging.info(f'The best miou during the training is {currentIou}')
            
    return

if __name__ == '__main__':
    main()