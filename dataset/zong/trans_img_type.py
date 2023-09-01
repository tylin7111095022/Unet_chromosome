import os
import cv2
import json
import numpy as np
from PIL import Image
import re

def main():
    # mask_normalize(root="predict") # 將mask轉成0和1
    # mask_unnormalize(root="mask") # 將mask轉成0和255
    # rename_for_image_labeler(root="predict", imgext = "jpg", pattern="Label_") #重新命名並傳回紀錄原本檔案名跟對應label名的json檔
    # rename_for_image_labeler(root="F:/2023/chromosomes/CenterNet2_docker/mono_chromosome", imgext = "jpg", pattern="Instance_")
    
    # 將Instance_?跟Label_?轉回原本的檔名
    revert_name(mask_dir="mask", json_file="Label_namepair.json")
    # revert_name(mask_dir="F:/2023/chromosomes/CenterNet2_docker/mono_chromosome", json_file="Instance_namepair.json")
    return
    
def mask_normalize(root:str):
    for dirname, subdirs, files in os.walk(root):
        for f in files:
            img_path = os.path.join(dirname,f)
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            img = img / 255
            img = img.astype(np.uint8)
            # print(img.shape)
            im=Image.fromarray(img)
            im.save(img_path)

def mask_unnormalize(root:str):
    for dirname, subdirs, files in os.walk(root):
        for f in files:
            img_path = os.path.join(dirname,f)
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            img = img * 255
            img = img.astype(np.uint8)
            # print(img.shape)
            im=Image.fromarray(img)
            im.save(img_path)

def rename_for_image_labeler(root:str, imgext = "jpg", pattern:str="Label_",):
    dict_name = {name:pattern+str(i+1) + "." + name.split(".")[-1] for i, name in enumerate(os.listdir(root)) if name.split(".")[-1]==imgext}

    for key in dict_name.keys():
        os.rename(os.path.join(root, key), os.path.join(root, dict_name[key]))

    with open(f"{pattern}namepair.json", "w") as outfile:
        json.dump(dict_name, outfile,indent=4)

    return dict_name

def revert_name(mask_dir, json_file):
    with open(json_file) as f:
        dictionary = json.load(f)
    label2orig = {k:v for v, k in dictionary.items()}
    masks = os.listdir(mask_dir)
    for key in label2orig.keys():
        os.rename(os.path.join(mask_dir, key), os.path.join(mask_dir, label2orig[key]))

    return

def transfer_file_type(source_folder, target_folder,source_extension=".png$", target_extension=".jpg"):
    """本函式用來將某資料夾內符合特定格式的檔案另存為另一格式的檔案並放在目標檔案夾下
    source_extension必須為正規表示式, .png$ 表示結尾需要符合.png"""
    # 如果target folder 不在，建立該資料夾
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # 遍歷sourcr folder中的所有檔案
    for filename in os.listdir(source_folder):
        if bool(re.search(source_extension, filename)):
            # 構建tif文件路径和目标jpg文件路徑
            src_path = os.path.join(source_folder, filename)
            tg_path = os.path.join(target_folder, os.path.splitext(filename)[0] + target_extension)

            # 打开原始图像文件
            src_image = Image.open(src_path)
            
            # 圖像儲存為想要的格式
            src_image.save(tg_path, 'JPEG')

if __name__ == "__main__":
    main()