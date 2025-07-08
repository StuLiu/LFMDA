"""
@Filename:
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""
import os
import time
import torch
import json
import cv2
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.nn.utils import clip_grad
from ever.core.iterator import Iterator

from lfmda.datasets import *
from lfmda.datasets.daLoader import DALoader
from lfmda.models import get_model
from lfmda.gast.balance import CrossEntropy
from lfmda.utils.eval import evaluate
from lfmda.utils.tools import *
from lfmda.utils.output_similarity import OutputCosineSimilarity


parser = argparse.ArgumentParser(description='Get image pairs similarity.')
parser.add_argument('--config-path', type=str, default='st.lfmda.2potsdam', help='config path')
parser.add_argument('--src-dir', default='data/IsprsDA/Vaihingen/img_dir/train')
parser.add_argument('--tgt-dir', default='data/IsprsDA/Potsdam/img_dir/train')
parser.add_argument('--save-path', default='data/IsprsDA/Potsdam/file_2vaihingen.json')
parser.add_argument('--topk', type=int, default=65535)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
cfg = import_config(args.config_path, create=False, copy=False)
cfg.SNAPSHOT_DIR = None


def main():
    time_from = time.time()

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    backbone = str(cfg.BACKBONE)
    model_name = str(cfg.MODEL)
    pretrained = str(cfg.PRETRAINED)
    stop_steps = 1000
    cfg.NUM_STEPS = stop_steps * 1.5
    cfg.PREHEAT_STEPS = int(stop_steps / 20)  # for warm-up

    # model for semantic segmentation
    model, feat_channel = get_model(class_num, model_name, backbone, pretrained)
    model = model.cuda()
    model.train()

    # loss function
    loss_fn_s = CrossEntropy(ignore_label=ignore_label, class_balancer=None)

    # dataloaders
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)

    epochs = stop_steps / len(sourceloader)
    logging.info(f'batch num: source={len(sourceloader)}')
    logging.info('epochs ~= %.3f' % epochs)

    # optimizer
    if cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(get_param_groups(model),
                                lr=cfg.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.WEIGHT_DECAY)

    # training
    for i_iter in tqdm(range(stop_steps)):

        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        optimizer.zero_grad()

        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        pred_s1, pred_s2, feat_s = model(images_s)

        # loss seg
        loss_seg = loss_calc([pred_s1, pred_s2], label_s, loss_fn=loss_fn_s, multi=True)
        loss_seg.backward(retain_graph=False)

        # gradient clip to avoid grad boomb
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=32, norm_type=2)
        optimizer.step()

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            loss = loss_seg
            logging.info(f'iter={i_iter + 1}, total={loss:.3f}, loss_seg={loss_seg:.3f}, lr={lr}')

        if (i_iter + 1) >= stop_steps or (i_iter + 1) % 500 == 0:
            _, miou_curr = evaluate(model, cfg, True, None)
            model.train()

    # compute similarity for cross-domain image pairs
    ccs = OutputCosineSimilarity(cfg, model, device='cuda')
    imgs_src = glob(str(args.src_dir) + r'/*.png')
    imgs_tgt = glob(str(args.tgt_dir) + r'/*.png')
    topk = min(len(imgs_src), args.topk)
    logging.info(f'src len={len(imgs_src)}, tgt len={len(imgs_tgt)}, topk={topk}')

    # preprocess encode for src images
    files_src = [osp.basename(img_src) for img_src in imgs_src]
    imgs_src_feat = []
    logging.info(f'encode source images ...')
    for img_src in tqdm(imgs_src):
        imgs_src_feat.append(ccs.encode(img_src))

    logging.info(f'compute similarity ...')
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

        if args.debug:
            img_tgt_npy = cv2.imread(img_tgt, flags=cv2.IMREAD_UNCHANGED)
            img_show = [img_tgt_npy]
            # cv2.imshow('img_tgt_npy', img_tgt_npy)
            for _, img_name_str_ in enumerate(topk_src_names[:2] + topk_src_names[-2:]):
                img_show.append(cv2.imread(osp.join(args.src_dir, img_name_str_)))
                # cv2.imshow(f'img_tgt_npy_{i}', img_src_npy)
            img_show = np.concatenate(img_show, axis=1)
            cv2.imshow(f'gt-top1-top2-topr2-topr1', img_show)
            if ord('q') == cv2.waitKey(0):
                exit(0)
        torch.cuda.empty_cache()

    with open(f'{args.save_path}', 'w') as of:
        json.dump(file_tgt2src, of, indent=4)

    logging.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
