"""
@Project : NonDA2
@File    : convert_fbp.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/7/24 下午9:33
@e-mail  : 1183862787@qq.com
"""
import os
import numpy as np
import shutil
from skimage import io
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


warnings.filterwarnings('ignore')
thread_num = 16
threadpool = ThreadPoolExecutor(thread_num)
all_tasks = []


def read_save(image_folder, label_folder, image_name, label_name, out_folder_img, out_folder_lbl,
                             block_size=1024, overlap=512):
    # Read image and label
    image = io.imread(str(os.path.join(image_folder, image_name)))[:, :, 1:]    # get R, G, B
    label = io.imread(str(os.path.join(label_folder, label_name)))
    height, width = image.shape[: 2]
    for y in range(0, height - block_size + 1, block_size - overlap):
        for x in range(0, width - block_size + 1, block_size - overlap):
            y_to, x_to = y + block_size, x + block_size
            block_image = image[y:y_to, x:x_to, :]
            block_label = label[y:y_to, x:x_to]

            block_image_path = os.path.join(out_folder_img, f"{image_name[:-4]}_{y}_{y_to}_{x}_{x_to}.png")
            block_label_path = os.path.join(out_folder_lbl, f"{image_name[:-4]}_{y}_{y_to}_{x}_{x_to}.png")

            print(f"{image_name[:-4]}_{y}_{y_to}_{x}_{x_to}.png")
            io.imsave(block_image_path, block_image)
            io.imsave(block_label_path, block_label)


# Example usage:
image_folder = "/home/liuwang/liuwang_data/documents/datasets/seg/Five-Billion-Pixels/Image__8bit_NirRGB"
label_folder = "/home/liuwang/liuwang_data/documents/datasets/seg/Five-Billion-Pixels/Annotation__index"
output_folder_img = "/home/liuwang/liuwang_data/documents/datasets/seg/Five-Billion-Pixels/mmseg/img_dir"
output_folder_lbl = "/home/liuwang/liuwang_data/documents/datasets/seg/Five-Billion-Pixels/mmseg/ann_dir"
os.makedirs(output_folder_img, exist_ok=True)
os.makedirs(output_folder_lbl, exist_ok=True)


def run():
    all_tasks.clear()
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    image_files.sort()
    label_files.sort()
    # Assuming images and labels are named the same
    for image_name, label_name in tqdm(zip(image_files, label_files)):
        # read_save(image_folder, label_folder, image_name, label_name,
        #           output_folder_img, output_folder_lbl, 1024, 512)
        if len(all_tasks) < thread_num:
            all_tasks.append(threadpool.submit(read_save, image_folder, label_folder, image_name, label_name,
                                               output_folder_img, output_folder_lbl, 1024, 512))
        if len(all_tasks) == thread_num:
            wait(all_tasks)
            all_tasks.clear()
    if len(all_tasks) > 0:
        wait(all_tasks)
        all_tasks.clear()

run()
# read_images_from_folders(image_folder, label_folder, output_folder_img, output_folder_lbl)


img_dir_train = f'{output_folder_img}/train'
img_dir_val = f'{output_folder_img}/val'
ann_dir_train = f'{output_folder_lbl}/train'
ann_dir_val = f'{output_folder_lbl}/val'
os.makedirs(img_dir_train, exist_ok=True)
os.makedirs(img_dir_val, exist_ok=True)
os.makedirs(ann_dir_train, exist_ok=True)
os.makedirs(ann_dir_val, exist_ok=True)


image_files = os.listdir(image_folder)
eval_prefix = []
for image_file in image_files:
    if np.random.random() > 0.95:
        eval_prefix.append(image_file[:-4])

for img in os.listdir(output_folder_img):
    if not img.endswith('.tif') and not img.endswith('.png'): continue
    training = True
    for pref in eval_prefix:
        if img.startswith(pref):
            training = False
            break
    if training:
        shutil.move(os.path.join(output_folder_img, img), os.path.join(img_dir_train, img))
        shutil.move(os.path.join(output_folder_lbl, img), os.path.join(ann_dir_train, img))
    else:
        shutil.move(os.path.join(output_folder_img, img), os.path.join(img_dir_val, img))
        shutil.move(os.path.join(output_folder_lbl, img), os.path.join(ann_dir_val, img))
