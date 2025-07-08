
import os
import cv2
import warnings
import numpy as np
import torch
import torch.nn as nn

from skimage.io import imsave, imread

from lfmda.datasets import *
from lfmda.utils.tools import *
from lfmda.models import get_model


class Segmenter(nn.Module):
    def __init__(self, config_path='st.regda.pRgb2hnu', ckpt_path='ckpts/Hnu_curr.pth', size=(512, 512), alpha=0.5):
        super().__init__()

        warnings.filterwarnings('ignore')
        self.size = size
        self.alpha = alpha
        self.cfg = import_config(config_path, copy=False, create=False)

        class_num = len(eval(self.cfg.DATASETS).LABEL_MAP)
        backbone = str(self.cfg.BACKBONE)
        model_name = str(self.cfg.MODEL)

        # model for semantic segmentation
        model, feat_channel = get_model(class_num, model_name, backbone, None)
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        self.model = model.cuda()
        self.model.eval()

        self.color_list = np.array(list(eval(self.cfg.DATASETS).COLOR_MAP.values()))
        self.color_tensor = torch.from_numpy(self.color_list).cuda()
        self.trans = self.cfg.TEST_DATA_CONFIG['transforms']

    def forward(self, img_ndarray) -> np.ndarray:
        """
        get painted image.
        Args:
            img_ndarray: (h, w, 3)

        Returns:
            img_color: (h, w, 3)
        """
        size_origin = img_ndarray.shape[:2]
        # norm
        img_resized = cv2.resize(img_ndarray, dsize=(self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        img = self.trans(image=img_resized)['image'].unsqueeze(dim=0).cuda()
        # infer
        with torch.no_grad():
            cls = self.model(img)
            cls = cls.argmax(dim=1).squeeze().cpu().numpy()
        # vis
        img_colorer = overlay_segmentation_cv2(img_resized, cls, self.color_list, self.alpha)
        img_colorer = cv2.resize(img_colorer, dsize=(size_origin[-1], size_origin[0]), interpolation=cv2.INTER_LINEAR)
        return img_colorer

    def forward_cuda(self, img_ndarray) -> np.ndarray:
        size_origin = img_ndarray.shape[:2]
        # norm
        img_resized = cv2.resize(img_ndarray, dsize=(self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        img = self.trans(image=img_resized)['image'].unsqueeze(dim=0).cuda()
        # infer
        with torch.no_grad():
            cls = self.model(img)
            cls = cls.argmax(dim=1)
        # vis
        mask_color = render_segmentation_cuda(cls, self.color_tensor).squeeze(dim=0).cpu().numpy()
        img_color = img_resized * self.alpha + mask_color * (1 - self.alpha)
        img_color = cv2.resize(img_color, dsize=(size_origin[-1], size_origin[0]), interpolation=cv2.INTER_LINEAR)
        return img_color.astype(np.uint8)


if __name__ == '__main__':
    import time
    segmenter = Segmenter(config_path='st.lfmda.uav2uav_sourceonly', ckpt_path='ckpts/SegFormer-b0-uav-best.pth',
                          size=(384, 512))
    img_rgb = imread('demo/frame_43.png')
    time_from = time.time()
    img_res = None
    for i in range(300):
        # img_res = segmenter.forward_cuda(img_rgb)  # 7s
        img_res = segmenter(img_rgb)   # 8s
    print(f'using {int(time.time() - time_from)} s')
    cv2.imshow('res', cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
