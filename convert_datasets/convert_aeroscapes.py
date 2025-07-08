"""
@Project : Nonda2
@File    : convert_aeroscapes.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/7/27 2:13
@e-mail  : 1183862787@qq.com
"""
import shutil
import os
from tqdm import tqdm


meta_train_path = 'G:\\datasets\\aeroscapes\\ImageSets\\trn.txt'
meta_val_path = 'G:\\datasets\\aeroscapes\\ImageSets\\val.txt'
src_img_dir = 'G:\\datasets\\aeroscapes\\JPEGImages'
src_ann_dir = 'G:\\datasets\\aeroscapes\\SegmentationClass'
src_vis_dir = 'G:\\datasets\\aeroscapes\\Visualizations'
img_dir_train = '..\\data\\aeroscapes\\img_dir\\train'
img_dir_val = '..\\data\\aeroscapes\\img_dir\\val'
ann_dir_train = '..\\data\\aeroscapes\\ann_dir\\train'
ann_dir_val = '..\\data\\aeroscapes\\ann_dir\\val'
vis_dir_train = '..\\data\\aeroscapes\\ann_dir_color\\train'
vis_dir_val = '..\\data\\aeroscapes\\ann_dir_color\\val'
os.makedirs(img_dir_train, exist_ok=True)
os.makedirs(img_dir_val, exist_ok=True)
os.makedirs(ann_dir_train, exist_ok=True)
os.makedirs(ann_dir_val, exist_ok=True)
os.makedirs(vis_dir_train, exist_ok=True)
os.makedirs(vis_dir_val, exist_ok=True)


with open(meta_train_path, 'r') as f:
    lines = f.readlines()
    train_img_names = [f'{line[:-1]}' for line in lines]
for train_img_name in tqdm(train_img_names):
    shutil.copyfile(f'{src_img_dir}\\{train_img_name}.jpg', f'{img_dir_train}\\{train_img_name}.jpg')
    shutil.copyfile(f'{src_ann_dir}\\{train_img_name}.png', f'{ann_dir_train}\\{train_img_name}.png')
    shutil.copyfile(f'{src_vis_dir}\\{train_img_name}.png', f'{vis_dir_train}\\{train_img_name}.png')

with open(meta_val_path, 'r') as f:
    lines = f.readlines()
    val_img_names = [f'{line[:-1]}' for line in lines]

for val_img_name in tqdm(val_img_names):
    shutil.copyfile(f'{src_img_dir}\\{val_img_name}.jpg', f'{img_dir_val}\\{val_img_name}.jpg')
    shutil.copyfile(f'{src_ann_dir}\\{val_img_name}.png', f'{ann_dir_val}\\{val_img_name}.png')
    shutil.copyfile(f'{src_vis_dir}\\{val_img_name}.png', f'{vis_dir_val}\\{val_img_name}.png')

