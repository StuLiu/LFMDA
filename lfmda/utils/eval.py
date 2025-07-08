import logging
import os
import cv2

from lfmda.datasets.daLoader import DALoader
from lfmda.utils.tools import *
from ever.util.param_util import count_model_parameters
from lfmda.viz import VisualizeSegmm
from argparse import ArgumentParser
from lfmda.datasets import *
from lfmda.gast.metrics import PixelMetricIgnore


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None, slide=True, tta=False, test=False, vis=True):
    logger = logger if logger else logging.getLogger()

    ignore_labels = []
    if cfg.DATASETS == 'IsprsDA':
        ignore_labels = [0]

    if cfg.SNAPSHOT_DIR is not None:
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        viz_op = VisualizeSegmm(vis_dir, eval(cfg.DATASETS).PALETTE)

    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()

    if test:
        eval_dataloader = DALoader(cfg.TEST_DATA_CONFIG, cfg.DATASETS)
    else:
        eval_dataloader = DALoader(cfg.EVAL_DATA_CONFIG, cfg.DATASETS)

    class_names = eval(cfg.DATASETS).COLOR_MAP.keys()
    num_class = len(class_names)

    metric_op = PixelMetricIgnore(len(class_names), class_names=list(class_names),
                                  logdir=cfg.SNAPSHOT_DIR, logger=logger,
                                  ignore_labels=ignore_labels)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.cuda()
            cls = pre_slide(model, ret, num_classes=num_class, tta=tta) if slide else model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()

            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0

            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)

            if cfg.SNAPSHOT_DIR is not None and vis:
                for fname, pred in zip(ret_gt['fname'], cls):
                    fname = fname.replace('tif', 'png')
                    fname = fname.replace('jpg', 'png')
                    viz_op(pred, fname)

    torch.cuda.empty_cache()
    return metric_op.summary_all()


def predict(model, cfg, is_training=False, ckpt_path=None, logger=None, slide=True, tta=False, test=False, vis=True):
    logger = logger if logger else logging.getLogger()

    if cfg.SNAPSHOT_DIR is not None:
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, f'vis-{os.path.basename(ckpt_path)}-{"test" if test else "val"}')
        viz_op = VisualizeSegmm(vis_dir, eval(cfg.DATASETS).PALETTE)

    out_dir = os.path.join(cfg.SNAPSHOT_DIR, f'ids-{os.path.basename(ckpt_path)}-{"test" if test else "val"}')
    os.makedirs(out_dir, exist_ok=True)

    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()

    if test:
        eval_dataloader = DALoader(cfg.TEST_DATA_CONFIG, cfg.DATASETS)
    else:
        eval_dataloader = DALoader(cfg.EVAL_DATA_CONFIG, cfg.DATASETS)

    class_names = eval(cfg.DATASETS).COLOR_MAP.keys()
    num_class = len(class_names)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.cuda()
            cls = pre_slide(model, ret, num_classes=num_class, tta=tta) if slide else model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()

            for fname, pred in zip(ret_gt['fname'], cls):
                fname = fname.replace('tif', 'png')
                fname = fname.replace('jpg', 'png')
                cv2.imwrite(f"{out_dir}/{fname}", pred.astype(np.uint8))
                if cfg.SNAPSHOT_DIR is not None and vis:
                    viz_op(pred, fname)

    torch.cuda.empty_cache()
