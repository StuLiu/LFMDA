import shutil
import warnings
import os

from argparse import ArgumentParser
from skimage.io import imsave, imread

from lfmda.datasets import *
from lfmda.utils.tools import *
from lfmda.viz import VisualizeSegmm
# from lfmda.models.Encoder import Deeplabv2
from lfmda.models import get_model


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('config_path', type=str, help='config path')
    parser.add_argument('ckpt_path', type=str, help='ckpt path')
    parser.add_argument('image_path', type=str, help='ckpt path')
    parser.add_argument('--save-dir', type=str, default='./demo', help='save dir')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--slide', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--tta', type=str2bool, default=False, help='save dir path')
    parser.add_argument('--gt', type=str2bool, default=False, help='save dir path')
    parser.add_argument('--offset', type=int, default=0, help='add to cls')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = import_config(args.config_path, copy=False, create=False)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    backbone = str(cfg.BACKBONE_STUDENT)
    model_name = str(cfg.MODEL_STUDENT)
    pretrained = str(cfg.PRETRAINED_STUDENT)

    # model for semantic segmentation
    model, feat_channel, _ = get_model(class_num, model_name, backbone, pretrained)
    model_state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)
    model = model.cuda()
    model.eval()

    viz_op = VisualizeSegmm(args.save_dir, eval(cfg.DATASETS).PALETTE)
    trans = cfg.TEST_DATA_CONFIG['transforms']
    with torch.no_grad():
        img_np = imread(args.image_path)
        img = trans(image=img_np)['image'].unsqueeze(dim=0).cuda()
        cls = pre_slide(model, img, num_classes=class_num, tta=args.tta) if args.slide else model(img)
        cls = cls.argmax(dim=1).cpu().numpy().squeeze()

        viz_op(cls, 'prediction_color.png')

        overlay = overlay_segmentation_cv2(img_np, cls, np.array(list(eval(cfg.DATASETS).COLOR_MAP.values())))
        imsave(os.path.join(args.save_dir, 'img_color.png'), overlay)

        imsave(os.path.join(args.save_dir, 'prediction.png'), (cls + args.offset).astype(np.uint8))

        if args.gt:
            gt_path = str(args.image_path).replace('img_dir', 'ann_dir')
            if os.path.exists(gt_path):
                print(gt_path)
                viz_op(imread(gt_path), f'gt.png')

    torch.cuda.empty_cache()

