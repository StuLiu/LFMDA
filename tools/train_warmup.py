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
from lfmda.loss import *
from lfmda.models import get_model, FCDiscriminator
from lfmda.datasets.daLoader import DALoader
from lfmda.datasets.nonblind import NonblindLoader
from lfmda.gast.alignment import DownscaleLabel


parser = argparse.ArgumentParser(description='Train in src.')
parser.add_argument('--config-path', type=str, default='st.lfmda.2vaihingen_segformer', help='config path')
parser.add_argument('--align-domain', action='store_true', help='whether align domain or not')
parser.add_argument('--loss-seg', type=str, default="CrossEntropy",
                    choices=['CrossEntropy', 'OhemCrossEntropy'], help='seg loss function')
parser.add_argument('--loss-kd', type=str, default="PrototypeContrastiveLoss",
                    choices=['MSELoss', 'PrototypeContrastiveLoss', 'None'], help='kd loss function')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

# get config from config.py
if args.loss_kd == 'MSELoss':
    postfix = f'/src_warmup_adaptseg_st-mse' if args.align_domain else f'/src_warmup_st-mse'
elif args.loss_kd == 'PrototypeContrastiveLoss':
    postfix = f'/src_warmup_adaptseg_st-pcd' if args.align_domain else f'/src_warmup_st-pcd'
else:
    postfix = f'/src_warmup_adaptseg_s' if args.align_domain else f'/src_warmup_s'
cfg = import_config(args.config_path, create=True, copy=True, postfix=postfix)


def main():
    time_from = time.time()

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name_tea = str(cfg.MODEL_TEACHER)
    model_name_stu = str(cfg.MODEL_STUDENT)
    backbone_tea = str(cfg.BACKBONE_TEACHER)
    backbone_stu = str(cfg.BACKBONE_STUDENT)
    pretrained_tea = str(cfg.PRETRAINED_TEACHER)
    pretrained_stu = str(cfg.PRETRAINED_STUDENT)
    stop_steps = cfg.STAGE1_STEPS
    cfg.NUM_STEPS = stop_steps * 1.5  # for learning rate poly

    # model for semantic segmentation
    model_tea, feat_channel, _ = get_model(class_num, model_name_tea, backbone_tea, pretrained_tea)
    model_stu, feat_channel, down_scale = get_model(class_num, model_name_stu, backbone_stu, pretrained_stu)
    model_tea = model_tea.cuda()
    model_stu = model_stu.cuda()
    model_tea.train()
    model_stu.train()
    # model_tea.freeze_backbone(freezing=True)
    freeze_model(model_tea.backbone, freezing=True)

    # discriminator
    model_d = FCDiscriminator(num_classes=class_num).cuda()

    # loss function
    loss_fn_seg = eval(args.loss_seg)(ignore_label=ignore_label, class_balancer=None)
    loss_fn_kd = get_kd_loss(loss_name=args.loss_kd, ignore_label=ignore_label, temperature=cfg.KD_TEM)

    # aligner
    need_init = isinstance(loss_fn_kd, PrototypeContrastiveLoss)
    aligner = Aligner(logger=logger,
                      feat_channels=feat_channel,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.99,
                      down_scale=down_scale,
                      resume=osp.join(cfg.SNAPSHOT_DIR, '..', 'prototypes/init_prototypes.pth') if need_init else None)

    # dataloaders
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)
    epochs = stop_steps / len(sourceloader)
    logger.info(f'batch num: source={len(sourceloader)}, target={len(targetloader)}')
    logger.info('epochs ~= %.3f' % epochs)

    # optimizer
    if cfg.OPTIMIZER == 'SGD':
        optimizer_tea = optim.SGD(model_tea.parameters(),
                                  lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        optimizer_stu = optim.SGD(model_stu.parameters(),
                                  lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer_tea = optim.AdamW(get_param_groups(model_tea), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        optimizer_stu = optim.AdamW(get_param_groups(model_stu), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    optimizer_d = optim.Adam(model_d.parameters(), lr=1e-4, betas=(0.9, 0.99))

    miou_max, iter_max, loss_kd, loss_domain = 0, 0, 0, 0

    # training
    for i_iter in tqdm(range(stop_steps)):

        adjust_learning_rate(optimizer_tea, i_iter, cfg)
        lr = adjust_learning_rate(optimizer_stu, i_iter, cfg)
        optimizer_tea.zero_grad()
        optimizer_stu.zero_grad()
        
        # get source training data
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        bs, in_c, h, w = images_s.shape

        # #####################
        # ### train for seg ###
        # #####################
        pred_s_tea1, pred_s_tea2, _ = model_tea(images_s)
        pred_s_stu1, pred_s_stu2, _ = model_stu(images_s)
        loss_seg_tea = loss_calc([pred_s_tea1, pred_s_tea2], label_s, loss_fn=loss_fn_seg, multi=True)
        loss_seg_stu = loss_calc([pred_s_stu1, pred_s_stu2], label_s, loss_fn=loss_fn_seg, multi=True)
        (loss_seg_tea + loss_seg_stu).backward(retain_graph=False)

        # ##########################################
        # #### train for knowledge distillation ####
        # ##########################################

        images_t = targetloader_iter.next()[0][0].cuda()
        images_st = torch.cat([images_s[:bs//2], images_t[:bs//2]], dim=0)
        pred_st_tea1, pred_st_tea2, feat_st_tea = model_tea(images_st)
        _, _, feat_st_stu = model_stu(images_st)

        # # resize feats
        feat_st_tea = tnf.interpolate(feat_st_tea, size=feat_st_stu.shape[-2:], mode='bilinear')

        # # comput kd loss
        if isinstance(loss_fn_kd, PrototypeContrastiveLoss):
            # # get pseudo labels
            pred_st_tea = (pred_st_tea1.softmax(dim=1) + pred_st_tea2.softmax(dim=1)) / 2
            max_pred, pseudo_label_st = torch.max(pred_st_tea, dim=1)
            pseudo_label_st[max_pred < 0.9] = ignore_label
            label_st = torch.cat([label_s[:bs//2], pseudo_label_st[bs//2:]], dim=0)
            # # update prototypes
            label_st_down = aligner.update_prototype(feat_st_tea.detach(), label_st)
            # # compute pcd loss
            loss_kd = loss_fn_kd(aligner.prototypes, feat_st_stu, label_st_down)
            loss_kd.backward(retain_graph=False)
        elif isinstance(loss_fn_kd, KDMSELoss):
            # # compute kd-mse loss
            loss_kd = loss_fn_kd(feat_st_tea, feat_st_stu, None)
            loss_kd.backward(retain_graph=False)
        
        if args.align_domain:
            # align source and target feature distributions in domain level

            # #################
            # train generator #
            # #################
            freeze_model(model_d, freezing=True)
            batch_size_align_domain = images_s.shape[0] // 2
            img_src = images_s[:batch_size_align_domain]
            img_tgt = targetloader_iter.next()[0][0][:batch_size_align_domain].cuda()
            _, pred_st, feats_st = model_stu(torch.cat([img_src, img_tgt], dim=0))
            d_out_st = model_d(torch.softmax(pred_st, dim=1))
            # # adaptSeg
            lbl_st = torch.zeros_like(d_out_st).cuda()
            lbl_st[: d_out_st.shape[0] // 2, :, :, :] = 1               # gt: src=0, tgt=1. here reversed
            loss_dist = bce_loss(d_out_st, lbl_st)                      # generate similar outputs

            (0.001 * loss_dist).backward(retain_graph=False)

            # #####################
            # train discriminator #
            # #####################
            # # adaptSeg
            adjust_learning_rate(optimizer_d, i_iter, cfg)
            optimizer_d.zero_grad()
            freeze_model(model_d, freezing=False)
            d_out_st = model_d(torch.softmax(pred_st.detach(), dim=1))
            lbl_st = 1 - lbl_st
            loss_adv = bce_loss(d_out_st, lbl_st)
            (0.001 * loss_adv).backward(retain_graph=False)

            loss_domain = loss_dist + loss_adv

        optimizer_tea.step()
        optimizer_stu.step()
        if args.align_domain:
            optimizer_d.step()

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            loss = loss_seg_tea + loss_seg_stu + loss_domain
            log_loss = f'iter={i_iter + 1}, total={loss:.3f}, loss_seg_tea={loss_seg_tea:.3f}, ' \
                       f'loss_seg_stu={loss_seg_stu:.3f}, loss_kd={loss_kd:.3e}, ' \
                       f'loss_domain={loss_domain:.3e}, lr={lr:.3e}'
            logger.info(log_loss)

        if (i_iter + 1) % cfg.EVAL_EVERY == 0 or (i_iter + 1) >= stop_steps:
            ckpt_path_tea = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_tea_curr.pth')
            ckpt_path_stu = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_stu_curr.pth')
            torch.save(model_tea.state_dict(), str(ckpt_path_tea))
            torch.save(model_stu.state_dict(), str(ckpt_path_stu))
            if (i_iter + 1) >= stop_steps:
                _, miou_curr_tea = evaluate(model_tea, cfg, True, ckpt_path_tea, logger, vis=False)
            _, miou_curr_stu = evaluate(model_stu, cfg, True, ckpt_path_stu, logger)
            if miou_max <= miou_curr_stu:
                miou_max = miou_curr_stu
                iter_max = i_iter + 1
                torch.save(model_tea.state_dict(), str(osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_tea_best.pth')))
                torch.save(model_stu.state_dict(), str(osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_stu_best.pth')))
                if osp.isdir(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best')):
                    shutil.rmtree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
                shutil.copytree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{os.path.basename(ckpt_path_stu)}'),
                                os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
            logger.info(f'Best model_stu in iter={iter_max}, best_mIoU={miou_max}.')
            model_tea.train()
            model_stu.train()
    # save prototypes
    torch.save(aligner.prototypes.cpu(), os.path.join(cfg.SNAPSHOT_DIR, '../prototypes/warmup_prototypes.pth'))
    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
