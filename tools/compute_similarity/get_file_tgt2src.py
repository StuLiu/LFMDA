"""
@Project : DAFormer-master
@File    : compute.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/28 下午2:27
@e-mail  : 1183862787@qq.com
"""

import argparse
import json

import torch

from lfmda.utils.clip_cosine_similarity import ClipCosineSimilarity, RemoteClipCosineSimilarity
import sys
from glob import glob
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--src-dir', default='data/IsprsDA/Potsdam/img_dir/train')
    parser.add_argument('--tgt-dir', default='data/IsprsDA/Vaihingen/img_dir/train')
    parser.add_argument('--save-path', default='data/IsprsDA/Vaihingen/file_2vaihingen.json')
    parser.add_argument('--topk', type=int, default=65535)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    print(args)
    ccs = ClipCosineSimilarity(device='cpu')
    # ccs = RemoteClipCosineSimilarity(device='cpu')
    imgs_src = glob(str(args.src_dir) + r'/*.png')
    imgs_tgt = glob(str(args.tgt_dir) + r'/*.png')
    topk = min(len(imgs_src), args.topk)
    print(f'src len={len(imgs_src)}, tgt len={len(imgs_tgt)}, topk={topk}')

    # preprocess encode for src images
    files_src = [osp.basename(img_src) for img_src in imgs_src]
    imgs_src_feat = []
    print(f'encode source images ...')
    for img_src in tqdm(imgs_src):
        imgs_src_feat.append(ccs.encode(img_src))

    print(f'compute similarity ...')
    file_tgt2src = {}
    for img_tgt in tqdm(imgs_tgt):
        img_tgt_name = osp.basename(img_tgt)
        img_tgt_feat = ccs.encode(img_tgt)

        temp_sims = []
        for idx, img_src in enumerate(imgs_src):
            similarity = ccs(img_tgt_feat.cuda(), imgs_src_feat[idx].clone().cuda()).cpu().item()
            temp_sims.append(similarity)

        temp_sims = torch.from_numpy(np.array(temp_sims)).cuda()
        topk_vals, topk_idxes = torch.topk(temp_sims, k=topk)
        topk_src_names = [str(files_src[idx]) for idx in topk_idxes]
        file_tgt2src[str(img_tgt_name)] = topk_src_names

        if args.vis:
            img_tgt_npy = cv2.imread(img_tgt, flags=cv2.IMREAD_UNCHANGED)
            cv2.imshow('img_tgt_npy', img_tgt_npy)
            for i, img_name_str_ in enumerate(file_tgt2src[str(img_tgt_name)][:3]):
                img_src_npy = cv2.imread(osp.join(args.src_dir, img_name_str_))
                cv2.imshow(f'img_tgt_npy_{i}', img_src_npy)
            if ord('q') == cv2.waitKey(0):
                exit(0)
        torch.cuda.empty_cache()

    with open(f'{args.save_path}', 'w') as of:
        json.dump(file_tgt2src, of, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])