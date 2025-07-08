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

from tqdm import tqdm
from ever.core.iterator import Iterator

from lfmda.datasets import *
from lfmda.gast.alignment import Aligner
from lfmda.models import get_model
from lfmda.utils.tools import *
from lfmda.utils.eval import evaluate


# CUDA_VISIBLE_DEVICES=7 python tools/init_prototypes.py --config-path st.proca.2potsdam --ckpt-model
# log/GAST/2potsdam/src/Potsdam_best.pth --ckpt-proto log/proca/2potsdam/align/prototypes_best.pth

parser = argparse.ArgumentParser(description='init proto')
parser.add_argument('--config-path', type=str, default='st.lfmda.2potsdam_segformer', help='config path')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path, create=True, copy=False, postfix=f'/prototypes')


def main():
    time_from = time.time()

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    backbone = str(cfg.BACKBONE_TEACHER)
    model_name = str(cfg.MODEL_TEACHER)
    pretrained = str(cfg.PRETRAINED_TEACHER)
    downscale = 16

    # model for semantic segmentation
    model, feat_channel, _ = get_model(class_num, model_name, backbone, pretrained)
    model = model.cuda()
    model = model.train()
    # model.freeze_backbone(freezing=True)
    freeze_model(model.backbone, freezing=True)
    logger.info(f'model_name={model_name}, feat_channel={feat_channel}, downscale={downscale}')

    aligner = Aligner(logger=logger,
                      feat_channels=feat_channel,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      down_scale=downscale,
                      decay=0.996)

    # source and target loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)

    for _ in tqdm(range(len(sourceloader))):
        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        with torch.no_grad():
            pred_s1, pred_s2, feat_s = model(images_s)

        # up-sampling
        h_origin, w_origin = label_s.shape[-2:]
        feat_s = tnf.interpolate(feat_s, size=(h_origin // downscale, w_origin // downscale), mode='bilinear')

        # avg-updating prototypes
        aligner.update_avg(feat_s, label_s)

    aligner.init_avg()
    torch.save(aligner.prototypes.cpu(), os.path.join(cfg.SNAPSHOT_DIR, 'init_prototypes.pth'))
    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
