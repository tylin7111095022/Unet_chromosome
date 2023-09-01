import cv2
import torch
import numpy as np

def clahe(imgs:torch.Tensor,clipLimit:float, tileGridSize:tuple,show_pic:bool = False) -> torch.Tensor:
    '''
    注意本函式使用的圖片必須是灰階\n
    img的維度為(B,C,H,W)
    global clahe
    '''
    assert imgs.dim() == 4 #確認維度為(B,C,H,W)
    img_a = torch.permute(imgs, dims=(0,2,3,1)).numpy()
    #print(f'shape of img_a {img_a.shape}')
    assert img_a.shape[3] == 1 #確認通道軸為一

    clahe_object = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize = tileGridSize )
    for i in range(img_a.shape[0]):
        if show_pic:
            cv2.imshow(f'orig{i}', img_a[i])
            cv2.waitKey(0)
        img_a = np.squeeze(img_a,axis=3) #去掉通道軸以利clahe object可使用
        img_a[i] = clahe_object.apply(img_a[i])
        img_a = np.expand_dims(img_a, axis=3) #把通道軸加回來
        if show_pic:
            cv2.imshow(f'global_clahe{i}', img_a[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    preprocess_imgs = torch.from_numpy(np.transpose(img_a,(0,3,1,2)))
    return preprocess_imgs


def patch_clahe(imgs:torch.Tensor,clipLimit:float, patch_tileGridSize:tuple=(16,16),show_pic:bool = False) -> torch.Tensor:
    f'''
    注意本函式使用的圖片必須是灰階\n
    img的維度為(B,C,H,W)
    local clahe
    '''
    assert imgs.dim() == 4
    img_a = torch.permute(imgs, dims=(0,2,3,1)).numpy()
    h,w = img_a.shape[1],img_a.shape[2]
    w_patch = w // 8
    h_patch = h // 8
    clahe_object = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize = patch_tileGridSize )
    for i in range(img_a.shape[0]):
        if show_pic:
            cv2.imshow(f'orig{i}', img_a[i])
            cv2.waitKey(0)
        pos_y = 0
        while pos_y < h:
            pos_x = 0
            while pos_x < w:
                patch = img_a[i,pos_y:pos_y+h_patch, pos_x:pos_x+w_patch, 0]
                img_a[i,pos_y:pos_y+h_patch, pos_x:pos_x+w_patch, 0] = clahe_object.apply(patch)
                pos_x += w_patch
            pos_y += h_patch

        if show_pic:
            cv2.imshow(f'local_clahe{i}', img_a[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    preprocess_imgs = torch.from_numpy(np.transpose(img_a,(0,3,1,2)))
    return preprocess_imgs


if __name__ =='__main__':
    img = cv2.imread('dataset\\testing\\images\\03.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_t = torch.from_numpy(img)
    orig_img = img_t.unsqueeze(dim=0).unsqueeze(dim=0) #加入通道軸及批次軸
    print(f'img_t shape: {img_t.shape}')
    #patch_clahe(orig_img, 2.0,(4,4),show_pic=True)
    clahe(orig_img, 2.0,(4,4),show_pic=True)
    

