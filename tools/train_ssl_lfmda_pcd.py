"""
@Filename:
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""

import os.path as osp
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.optim as optim

from tqdm import tqdm
from torch.nn.utils import clip_grad
from ever.core.iterator import Iterator

from lfmda.utils.eval import evaluate
from lfmda.utils.tools import *
from lfmda.models import get_model
from lfmda.datasets import *
from lfmda.gast.alignment import Aligner
from lfmda.gast.balance import CrossEntropy
from lfmda.gast.pseudo_generation import gener_target_pseudo, pseudo_selection
from lfmda.utils.local_region_homog import Homogenizer
from lfmda.loss import get_kd_loss


parser = argparse.ArgumentParser(description='Run lfmda methods. ssl')
parser.add_argument('--config-path', type=str, default='st.lfmda.2vaihingen_segformer',
                    help='config path')
# ckpts
parser.add_argument('--ckpt-model-tea', type=str,
                    default='log/lfmda/SegFormer_MiT-B2/2vaihingen/src_warmup/Vaihingen_tea_curr.pth',
                    help='teacher model ckpt from stage1')
parser.add_argument('--ckpt-model-stu', type=str,
                    default='log/lfmda/SegFormer_MiT-B2/2vaihingen/src_warmup/Vaihingen_stu_curr.pth',
                    help='student model ckpt from stage1')
parser.add_argument('--ckpt-proto', type=str,
                    default='log/lfmda/SegFormer_MiT-B2/2vaihingen/prototypes/warmup_prototypes.pth',
                    help='proto ckpt from stage1')
parser.add_argument('--gen', type=str2bool, default=1, help='if generate pseudo-labels')
# LRH
parser.add_argument('--refine-label', type=str2bool, default=1, help='whether refine the pseudo label')
parser.add_argument('--refine-mode', type=str, default='all', choices=['all'],
                    help='refine by prototype, label, or both')
parser.add_argument('--refine-temp', type=float, default=2.0, help='whether refine the pseudo label')
parser.add_argument('--sam-refine', action='store_true', help='whether LRH')
parser.add_argument('--percent', type=float, default=0.95, help='class-freq threshold for LRH')
args = parser.parse_args()

# get config from config.py
postfix = f'/ssl_pcd' + ('_proto' if args.refine_label else '')
postfix = postfix + (f'_sam_{args.percent}' if args.sam_refine else '')
cfg = import_config(args.config_path, create=True, copy=True, postfix=postfix)
print('args.sam_refine,', args.sam_refine)


def main():
    time_from = time.time()
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')
    os.makedirs(save_pseudo_label_path, exist_ok=True)

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
    stop_steps = cfg.STAGE2_STEPS
    cfg.NUM_STEPS = stop_steps * 1.05  # for learning rate poly
    cfg.PREHEAT_STEPS = 1500  # for warm-up

    # model for semantic segmentation
    model_tea, feat_channel, _ = get_model(class_num, model_name_tea, backbone_tea, None)
    model_stu, feat_channel, down_scale = get_model(class_num, model_name_stu, backbone_stu, None)
    ckpt_model_tea = torch.load(args.ckpt_model_tea, map_location=torch.device('cpu'))
    ckpt_model_stu = torch.load(args.ckpt_model_stu, map_location=torch.device('cpu'))
    model_tea.load_state_dict(ckpt_model_tea)
    model_stu.load_state_dict(ckpt_model_stu)
    model_tea = model_tea.cuda()
    model_stu = model_stu.cuda()
    model_tea.train()
    model_stu.train()
    model_tea.freeze_backbone(freezing=True)

    aligner = Aligner(logger=logger,
                      feat_channels=feat_channel,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.99,
                      down_scale=down_scale,
                      resume=args.ckpt_proto)

    homogenizer = Homogenizer(percent=args.percent, class_num=class_num, ignore_label=ignore_label)

    loss_fn_seg = CrossEntropy(ignore_label=ignore_label, class_balancer=None)
    loss_fn_kd = get_kd_loss(loss_name='PrototypeContrastiveLoss', ignore_label=ignore_label)

    # source loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    # pseudo loader (target)
    pseudo_loader = DALoader(cfg.PSEUDO_DATA_CONFIG, cfg.DATASETS)
    # target loader
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)
    logger.info(f'batch num: source={len(sourceloader)}, target={len(targetloader)}, pseudo={len(pseudo_loader)}')
    # print(len(targetloader))
    epochs = stop_steps / len(sourceloader)
    logger.info('epochs ~= %.3f' % epochs)

    miou_max, iter_max, loss_kd = 0, 0, 0

    # optimizer
    if cfg.OPTIMIZER == 'SGD':
        optimizer_tea = optim.SGD(model_tea.parameters(),
                                  lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        optimizer_stu = optim.SGD(model_stu.parameters(),
                                  lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer_tea = optim.AdamW(get_param_groups(model_tea), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        optimizer_stu = optim.AdamW(get_param_groups(model_stu), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    for i_iter in tqdm(range(stop_steps)):

        _ = adjust_learning_rate(optimizer_tea, i_iter, cfg)
        lr = adjust_learning_rate(optimizer_stu, i_iter, cfg)
        optimizer_tea.zero_grad()
        optimizer_stu.zero_grad()

        # Generate pseudo label
        # if i_iter % cfg.GENE_EVERY == 0:
        if i_iter == 0:
            if args.gen:
                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                gener_target_pseudo(cfg, model_tea, pseudo_loader, save_pseudo_label_path,
                                    size=eval(cfg.DATASETS).SIZE, save_prob=True, slide=True, ignore_label=ignore_label)
            target_config = cfg.TARGET_DATA_CONFIG
            target_config['mask_dir'] = [save_pseudo_label_path]
            logger.info(target_config)
            targetloader = DALoader(target_config, cfg.DATASETS)
            targetloader_iter = Iterator(targetloader)
            logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))
        torch.cuda.empty_cache()

        model_tea.train()
        # source input
        batch_s = sourceloader_iter.next()
        images_s, label_s = batch_s[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        # target input
        batch_t = targetloader_iter.next()
        images_t, label_t = batch_t[0]
        images_t, label_t_soft, regs_t = images_t.cuda(), label_t['cls'].cuda(), label_t['sup'].cuda()

        bs, in_c, h, w = images_s.shape

        # model forward
        # # source
        pred_s_tea1, pred_s_tea2, feat_s_tea = model_tea(images_s)
        pred_s_stu1, pred_s_stu2, feat_s_stu = model_stu(images_s)
        # # target
        pred_t_tea1, pred_t_tea2, feat_t_tea = model_tea(images_t)
        pred_t_stu1, pred_t_stu2, feat_t_stu = model_stu(images_t)

        # up-sampling
        feat_s_tea = tnf.interpolate(feat_s_tea, size=(h // down_scale, w // down_scale), mode='bilinear')
        feat_t_tea = tnf.interpolate(feat_t_tea, size=(h // down_scale, w // down_scale), mode='bilinear')

        # refine target soft labels by prototypes and predictions
        label_t_soft = aligner.label_refine(None, feat_t_tea,
                                            [pred_t_tea1, pred_t_tea2],
                                            label_t_soft,
                                            refine=args.refine_label,
                                            mode=args.refine_mode,
                                            temp=args.refine_temp)
        label_t_hard = pseudo_selection(label_t_soft, cutoff_top=cfg.CUTOFF_TOP, cutoff_low=cfg.CUTOFF_LOW,
                                        return_type='tensor', ignore_label=ignore_label)

        # LRH
        if args.sam_refine:
            label_t_hard = homogenizer(label_t_hard, regs_t.squeeze(dim=1))

        label_s_down = aligner.update_prototype(feat_s_tea, label_s)
        label_t_hard_down = aligner.update_prototype(feat_t_tea, label_t_hard)

        # ##########################################
        # ####      train for segmentation      ####
        # ##########################################
        loss_seg_s_tea = loss_calc([pred_s_tea1, pred_s_tea2], label_s, loss_fn=loss_fn_seg, multi=True)
        loss_seg_s_stu = loss_calc([pred_s_stu1, pred_s_stu2], label_s, loss_fn=loss_fn_seg, multi=True)
        loss_seg_t_tea = loss_calc([pred_t_tea1, pred_t_tea2], label_t_hard, loss_fn=loss_fn_seg, multi=True)
        loss_seg_t_stu = loss_calc([pred_t_stu1, pred_t_stu2], label_t_hard, loss_fn=loss_fn_seg, multi=True)
        loss_seg_tea = loss_seg_s_tea + loss_seg_t_tea
        loss_seg_stu = loss_seg_s_stu + loss_seg_t_stu
        loss = loss_seg_tea + loss_seg_stu

        # ##########################################
        # #### train for knowledge distillation ####
        # ##########################################
        if i_iter % cfg.KD_INTERVAL == 0:
            feat_st_stu = torch.cat([feat_s_stu, feat_t_stu], dim=0)
            label_st_down = torch.cat([label_s_down, label_t_hard_down], dim=0)
            loss_kd = loss_fn_kd(aligner.prototypes, feat_st_stu, label_st_down)
            loss += loss_kd

        loss.backward()

        optimizer_tea.step()
        optimizer_stu.step()

        loss_seg = loss_seg_tea + loss_seg_stu
        log_loss = (f'iter={i_iter + 1}, loss_seg={loss_seg:.3f}, '
                    f'loss_seg_s_tea={loss_seg_s_tea:.3f}, loss_seg_s_stu={loss_seg_s_stu:.3f}, '
                    f'loss_seg_t_tea={loss_seg_t_tea:.3f}, loss_seg_t_stu={loss_seg_t_stu:.3f}, '
                    f'loss_kd={loss_kd:.3e}lr = {lr:.3e}')

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            logger.info(log_loss)

        if (i_iter + 1) % cfg.EVAL_EVERY == 0 or (i_iter + 1) >= stop_steps or i_iter == 0:
            ckpt_path_tea = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_tea_curr.pth')
            ckpt_path_stu = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_stu_curr.pth')
            torch.save(model_tea.state_dict(), str(ckpt_path_tea))
            torch.save(model_stu.state_dict(), str(ckpt_path_stu))
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

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')
    shutil.rmtree(save_pseudo_label_path, ignore_errors=True)
    logger.info('removing pseudo labels')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
