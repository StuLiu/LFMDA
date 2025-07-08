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
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim

from lfmda.utils.eval import evaluate

from tqdm import tqdm
from torch.nn.utils import clip_grad
from ever.core.iterator import Iterator
from lfmda.datasets import *
from lfmda.gast.alignment import Aligner
from lfmda.gast.balance import *
from lfmda.utils.tools import *
from lfmda.models import get_model, FCDiscriminator
from lfmda.datasets.daLoader import DALoader
from lfmda.datasets.nonblind import NonblindLoader

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='Train in src.')
parser.add_argument('--config-path', type=str, default='st.proca.2vaihingen', help='config path')
parser.add_argument('--ls', type=str, default="CrossEntropy",
                    choices=['CrossEntropy', 'OhemCrossEntropy'], help='source loss function')
parser.add_argument('--bcs', type=str2bool, default=0, help='whether balance class for source')
parser.add_argument('--class-temp', type=float, default=2.0, help='smooth factor')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path, create=True, copy=True, postfix='')


def main():
    time_from = time.time()

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    backbone = str(cfg.BACKBONE)
    model_name = str(cfg.MODEL)
    pretrained = str(cfg.PRETRAINED)
    stop_steps = cfg.STAGE1_STEPS
    cfg.NUM_STEPS = stop_steps + 1  # for learning rate poly
    cfg.PREHEAT_STEPS = 1500  # for warm-up

    # model for semantic segmentation
    model, feat_channel, _ = get_model(class_num, model_name, backbone, pretrained)
    model = model.cuda()
    model.train()

    # class balancer for loss
    class_balancer_s = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)

    # loss function
    loss_fn_s = eval(args.ls)(ignore_label=ignore_label, class_balancer=class_balancer_s if args.bcs else None)

    # dataloaders
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)

    epochs = stop_steps / len(sourceloader)
    logger.info(f'batch num: source={len(sourceloader)}')
    logger.info('epochs ~= %.3f' % epochs)

    # optimizer
    if cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(get_param_groups(model),
                                lr=cfg.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.WEIGHT_DECAY)

    mIoU_max, iter_max = 0, 0
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

        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=32, norm_type=2)
        optimizer.step()

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            loss = loss_seg
            log_loss = f'iter={i_iter + 1}, total={loss:.3f}, loss_seg={loss_seg:.3f}, lr={lr:.3e}'
            logger.info(log_loss)
            if args.bcs:
                logger.info(str(loss_fn_s.class_balancer))

        if (i_iter + 1) % cfg.EVAL_EVERY == 0 or (i_iter + 1) >= stop_steps:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_curr.pth')
            torch.save(model.state_dict(), ckpt_path)
            _, mIoU_curr = evaluate(model, cfg, True, ckpt_path, logger)
            if mIoU_max <= mIoU_curr:
                mIoU_max = mIoU_curr
                iter_max = i_iter + 1
                torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_best.pth'))
                if osp.isdir(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best')):
                    shutil.rmtree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
                shutil.copytree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{os.path.basename(ckpt_path)}'),
                                os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
            logger.info(f'Best model in iter={iter_max}, best_mIoU={mIoU_max}.')
            model.train()

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
