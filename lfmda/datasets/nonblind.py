"""
@Project : rads2
@File    : DALoader.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/6/6 上午11:00
@e-mail  : 1183862787@qq.com
"""

import torch
import glob
import os
import logging
import json
import cv2
import numpy as np
import os.path as osp

from skimage.io import imread
from typing import List
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from ever.interface import ConfigurableMixin
from ever.api.data import CrossValSamplerGenerator

from lfmda.datasets import LoveDA, IsprsDA


logger = logging.getLogger(__name__)


class NonblindDatasets(Dataset):
    def __init__(self, config_src, datasets_src, config_tgt, datasets_tgt, topk):
        super().__init__()
        self.topk = topk
        self.datasets_src = eval(datasets_src)(
            image_dir=config_src['image_dir'],
            mask_dir=config_src['mask_dir'],
            transforms=config_src['transforms'],
            read_sup=False
        )
        mask_dir_tgt = None if not isinstance(config_tgt['image_dir'], List) else [None] * len(config_tgt['image_dir'])
        self.datasets_tgt = eval(datasets_tgt)(
            image_dir=config_tgt['image_dir'],
            mask_dir=mask_dir_tgt,
            transforms=config_tgt['transforms'],
            read_sup=False,
            label_type='prob'
        )
        logger.info(f"{config_src['data_root']}, {len(self.datasets_src)}")
        logger.info(f"{config_tgt['data_root']}, {len(self.datasets_tgt)}")
        logger.info(f"topk={topk}")

        def get_file_tgt2src(src_root, tgt_root, topk_):
            src_names = ['vaihingen', 'potsdam', 'gta', 'synthia']
            src_name, src_root_low = None, str(src_root).lower()
            for _name in src_names:
                if _name in src_root_low:
                    src_name = _name
            assert src_name is not None, f'No src dataset matched in {src_names} for {src_root_low}'

            with open(osp.join(tgt_root, f'file_2{src_name}.json'), 'r') as of:
                res = json.load(of)
            res_new = {k: v[:topk_] for k, v in res.items()}
            res_new_rev = {k: v[-topk_:] for k, v in res.items()}
            return res_new, res_new_rev

        # self.file_tgt2src={tgt_file: src_file(most similar)}
        # self.file_tgt2src_r={tgt_file: src_file(most dissimilar)}
        self.file_tgt2src, self.file_tgt2src_r = get_file_tgt2src(
            src_root=config_src['data_root'],
            tgt_root=config_tgt['data_root'],
            topk_=topk
        )
        self.file_src2tgt, self.file_src2tgt_r = get_file_tgt2src(
            src_root=config_tgt['data_root'],
            tgt_root=config_src['data_root'],
            topk_=topk
        )
        pass

    def _get_nonblind_src_img(self, file_tgt):
        candidates = self.file_tgt2src[file_tgt]
        candidates_r = self.file_tgt2src_r[file_tgt]
        file_src = np.random.choice(candidates)
        file_src_r = np.random.choice(candidates_r)
        # print(f'candidates len={len(candidates)}')
        # img_src = cv2.imread(f'data/IsprsDA/Potsdam/img_dir/train/{file_src}', flags=cv2.IMREAD_UNCHANGED)
        # img_tgt = cv2.imread(f'data/IsprsDA/Vaihingen/img_dir/train/{file_tgt}', flags=cv2.IMREAD_UNCHANGED)
        # cv2.imshow('img_src', img_src)
        # cv2.imshow('img_tgt', img_tgt)
        # cv2.waitKey(0)

        src_idx = self.datasets_src.file_to_idx[file_src]
        src_idx_r = self.datasets_src.file_to_idx[file_src_r]
        img_src, _ = self.datasets_src[src_idx]
        img_src_r, _ = self.datasets_src[src_idx_r]
        return img_src, img_src_r

    def _get_nonblind_tgt_img(self, file_src):
        candidates = self.file_src2tgt[file_src]
        candidates_r = self.file_src2tgt_r[file_src]
        file_tgt = np.random.choice(candidates)
        file_tgt_r = np.random.choice(candidates_r)

        # img_src = cv2.imread(f'data/IsprsDA/Potsdam/img_dir/train/{file_src}', flags=cv2.IMREAD_UNCHANGED)
        # img_tgt = cv2.imread(f'data/IsprsDA/Vaihingen/img_dir/train/{file_tgt}', flags=cv2.IMREAD_UNCHANGED)
        # cv2.imshow('img_src', img_src)
        # cv2.imshow('img_tgt', img_tgt)
        # cv2.waitKey(0)

        tgt_idx = self.datasets_tgt.file_to_idx[file_tgt]
        tgt_idx_r = self.datasets_tgt.file_to_idx[file_tgt_r]
        img_tgt, _ = self.datasets_tgt[tgt_idx]
        img_tgt_r, _ = self.datasets_tgt[tgt_idx_r]
        return img_tgt, img_tgt_r

    # def __getitem__(self, idx):
    #     img_tgt, lbl_info_tgt = self.datasets_tgt[idx % len(self.datasets_tgt)]
    #     img_src = self._get_nonblind_src_img(lbl_info_tgt['fname'])
    #     return img_src, img_tgt

    def __getitem__(self, idx):
        """
        Args:
            idx: random sampled index
        Returns:
            ancher:
            pos:
            neg:
        """
        p_tgt = len(self.datasets_tgt) * 1.0 / (len(self.datasets_src) + len(self.datasets_tgt))
        if np.random.choice([True, False], p=(p_tgt, 1 - p_tgt)):
            # get nonblind src image for target_img
            img_tgt, lbl_info_tgt = self.datasets_tgt[idx % len(self.datasets_tgt)]
            img_src, img_src_r = self._get_nonblind_src_img(lbl_info_tgt['fname'])
            return img_src, img_tgt, img_src_r
        else:
            img_src, lbl_info_src = self.datasets_src[idx // len(self.datasets_tgt)]
            img_tgt, img_tgt_r = self._get_nonblind_tgt_img(lbl_info_src['fname'])
            return img_tgt, img_src, img_tgt_r

    def __len__(self):
        return len(self.datasets_src) * len(self.datasets_tgt)


class NonblindLoader(DataLoader):
    def __init__(self, config_src, datasets_src, config_tgt, datasets_tgt, topk=64):
        dataset = NonblindDatasets(config_src, datasets_src, config_tgt, datasets_tgt, topk)
        sampler = RandomSampler(dataset)
        # print(config_src['batch_size'] // 2)
        super(NonblindLoader, self).__init__(
            dataset,
            config_src['batch_size'] // 2,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
