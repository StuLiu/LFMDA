import os
import logging

from ever.util.param_util import count_model_parameters

from argparse import ArgumentParser
from lfmda.datasets.daLoader import DALoader
from lfmda.utils.tools import *
from lfmda.viz import VisualizeSegmm
from lfmda.datasets import *
from lfmda.models import get_model
from lfmda.gast.metrics import PixelMetricIgnore
from lfmda.utils.eval import evaluate


if __name__ == '__main__':

    seed_torch(2333)

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str,
                        default='st.lfmda.2potsdam_segformer',
                        help='config path')
    parser.add_argument('--ckpt-path', type=str,
                        default='log/lfmda/SegFormer_MiT-B2/2potsdam/ssl_proto_sam_0.95/Potsdam_stu_best.pth',
                        help='ckpt path')
    parser.add_argument('--test', type=str2bool, default=True, help='evaluate the test/val set')
    parser.add_argument('--tta', type=str2bool, default=False, help='test time augmentation')
    args = parser.parse_args()

    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = os.path.dirname(args.ckpt_path)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='Baseline', logdir=log_dir)

    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name_stu = str(cfg.MODEL_STUDENT)
    backbone_stu = str(cfg.BACKBONE_STUDENT)
    pretrained_stu = str(cfg.PRETRAINED_STUDENT)

    # model for semantic segmentation
    model_stu, feat_channel, down_scale = get_model(class_num, model_name_stu, backbone_stu, args.ckpt_path)
    model_stu = model_stu.cuda()

    evaluate(model_stu, cfg, False, args.ckpt_path, logger, tta=args.tta, test=args.test)
