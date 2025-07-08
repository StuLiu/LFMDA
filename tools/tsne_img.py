"""
@Project : rads2
@File    : tsne_img.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/10 下午10:23
@e-mail  : 1183862787@qq.com
"""

"""
@Project : rads2
@File    : tsne2.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/4/30 下午5:32
@e-mail  : 1183862787@qq.com
"""
import logging
import argparse
import os
import random

import cv2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from numpy import reshape
from keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.manifold import TSNE
from ever.core.iterator import Iterator

from lfmda.datasets.daLoader import DALoader
from lfmda.utils.tools import *
from lfmda.datasets import *
from lfmda.models.Encoder import Deeplabv2
from lfmda.gast.alignment import DownscaleLabel

from tsne_dataset import TSNECrossDomain


if __name__ == '__main__':
    # hello_world()
    seed_torch(2333)

    parser = argparse.ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str, default='st.regda.2potsdam_tsne', help='config path')
    parser.add_argument('--ckpt-path-src', type=str, default='log/cutmix/2potsdam/src/Potsdam_best.pth', help='ckpt path')
    parser.add_argument('--ckpt-path-da', type=str, default='log/regda/2potsdam/align/Potsdam_best.pth', help='ckpt path')
    parser.add_argument('--multi-layer', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='is instance norm in net end?')
    args = parser.parse_args()

    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = os.path.join('log/tsne')
    os.makedirs(log_dir, exist_ok=True)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='TSNE', logdir=log_dir)

    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    model_src = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=args.multi_layer,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=args.ins_norm
    ))
    ckpt_model = torch.load(args.ckpt_path_src, map_location=torch.device('cpu'))
    model_src.load_state_dict(ckpt_model)
    model_src = model_src.cuda()

    model_da = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=args.multi_layer,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=args.ins_norm
    ))
    ckpt_model = torch.load(args.ckpt_path_da, map_location=torch.device('cpu'))
    model_da.load_state_dict(ckpt_model)
    model_da = model_da.cuda()


    tsne_ = TSNECrossDomain(logger=logger, n_class=class_num, show_class=[1, 2, 3])

    tsne_.plot_an_image(
        img_path_src='./data/IsprsDA/Vaihingen/img_dir/test/area2_0_2048_512_2560.png',
        label_path_src='./data/IsprsDA/Vaihingen/ann_dir/test/area2_0_2048_512_2560.png',
        img_path_tgt='./data/IsprsDA/Potsdam/img_dir/test/3_13_1024_1536_1536_2048.png',
        label_path_tgt='./data/IsprsDA/Potsdam/ann_dir/test/3_13_1024_1536_1536_2048.png',
        model=model_src, max_pixel=200, save_dir=log_dir, da=False)
    tsne_.plot_an_image(
        img_path_src='./data/IsprsDA/Vaihingen/img_dir/test/area2_0_2048_512_2560.png',
        label_path_src='./data/IsprsDA/Vaihingen/ann_dir/test/area2_0_2048_512_2560.png',
        img_path_tgt='./data/IsprsDA/Potsdam/img_dir/test/3_13_1024_1536_1536_2048.png',
        label_path_tgt='./data/IsprsDA/Potsdam/ann_dir/test/3_13_1024_1536_1536_2048.png',
        model=model_da, max_pixel=200, save_dir=log_dir, da=True)
