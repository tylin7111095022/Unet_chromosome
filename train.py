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

#custom module
from metric import compute_mIoU
from models import get_models
from dataset import GrayDataset

dir_img = r'dataset\zong\val2017' #訓練集的圖片所在路徑
dir_truth = r'dataset\zong\val_mask' #訓練集的真實label所在路徑
dir_checkpoint = '.\\' #儲存模型的權重檔所在路徑

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--lr','-r',type = float, default=2e-2,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    parser.add_argument('--model', type=str,default='bn_unet',help='models, option: bn_unet, in_unet')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset = GrayDataset(img_dir = dir_img, mask_dir= dir_truth)
    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(dir_checkpoint,"log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    model = get_models(model_name=args.model,args=args)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas=(0.9,0.999))
    criterion = torch.nn.CrossEntropyLoss()
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    Parameters of training:
        model: {args.model}
        in_channel: {args.in_channel}
        output_map: {args.classes}
        learning rate: {args.lr}
        optimizer : Adam
        loss function : CrossEntropy
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
            is_valid: bool = True):

    arg_loader = dict(batch_size = batch_size, num_workers = 0)
    if is_valid:
        n_train = int(0.8*len(dataset))
        n_valid = len(dataset) - n_train
        trainset, validset = random_split(dataset,[n_train, n_valid],generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(trainset,shuffle = False, **arg_loader)
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
        for imgs, truthes in tqdm(train_loader):
            imgs = imgs.to(dtype=torch.float32,device=device)
            truthes = truthes.squeeze(1).to(dtype=torch.float32,device = device)
            # print(imgs.shape, truthes.shape)
            logits = net(imgs) #Size([B, 1, 512, 512]
            # print(logits.shape, truthes.shape)
            loss = loss_fn(logits, truthes.long())
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
                    mask_pred = torch.argmax(torch.softmax(logits,dim=1),dim=1).to(torch.int64) # (batch, channel,h ,w)
                    #compute the IOU          
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