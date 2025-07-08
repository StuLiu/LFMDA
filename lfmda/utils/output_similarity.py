"""
@Project : Nonda2
@File    : output_similarity.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/8/1 0:57
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from skimage.io import imread
from lfmda.utils.tools import index2onehot, import_config
from lfmda.models import get_model


def get_category_freq(logits_img: torch.Tensor):
    assert logits_img.dim() == 4 and logits_img.size(0) == 1, 'shape must be (1, c, h, w)'
    pred = torch.argmax(logits_img, dim=1)  # (1, h, w)
    pred_onehot = index2onehot(pred, class_num=logits_img.size(1))  # (1, c, h, w)
    category_cnt = torch.sum(pred_onehot, dim=[0, 2, 3])
    category_freq = category_cnt / (torch.sum(category_cnt) + 1e-5)
    return category_freq


def get_output_similarity(logits_src, logits_tgt):
    assert logits_src.shape == logits_tgt.shape, f'logits_src and logits_tgt must have the same shape'
    cf_src = get_category_freq(logits_src)
    cf_tgt = get_category_freq(logits_tgt)
    simi = tnf.cosine_similarity(cf_src, cf_tgt, dim=0)
    return simi


class OutputCosineSimilarity:
    def __init__(self, cfg, model, device='cpu'):
        self.device = device
        self.model = model.to(self.device).eval()
        self.trans = cfg.TEST_DATA_CONFIG['transforms']

    def encode(self, img_path) -> torch.Tensor:
        """encode image to a category frequency vector"""
        with torch.no_grad():
            img_numpy = imread(img_path)
            img_tensor = self.trans(image=img_numpy)['image'].unsqueeze(dim=0).to(self.device)
            logits = self.model(img_tensor)
            category_freq = get_category_freq(logits)
            if self.device != 'cpu':
                category_freq = category_freq.cpu()
            return category_freq

    def __call__(self, tensors1, tensors2):
        simi_mat = torch.cosine_similarity(tensors1, tensors2, dim=0)
        return simi_mat


class OutputCosineSimilarityFromCkpt:
    def __init__(self, config='st.lfmda.2potsdam', ckpt_path='ckpts/Vaihingen_best.pth', device='cpu'):
        self.device = device

        cfg = import_config(config, copy=False, create=False)
        class_num = len(eval(cfg.DATASETS).LABEL_MAP)
        backbone = str(cfg.BACKBONE)
        model_name = str(cfg.MODEL)

        self.model = get_model(class_num=class_num, model_name=model_name, backbone_name=backbone)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)
        self.model = self.model.to(self.device).eval()
        self.trans = cfg.TEST_DATA_CONFIG['transforms']

    def encode(self, img_path) -> torch.Tensor:
        """encode image to a category frequency vector"""
        with torch.no_grad():
            img_numpy = imread(img_path)
            img_tensor = trans(image=img_numpy)['image'].unsqueeze(dim=0).to(self.device)
            logits = self.model(img_tensor)
            category_freq = get_category_freq(logits)
            return category_freq

    def __call__(self, tensors1, tensors2):
        simi_mat = torch.cosine_similarity(tensors1, tensors2, dim=0)
        return simi_mat


if __name__ == '__main__':
    logits1 = torch.zeros([1, 7, 6, 6])
    logits1[:, 2, :, :] = 1
    logits1[:, 3, :, 4:] = 2
    print(get_category_freq(logits1))
    print(get_output_similarity(logits1, logits1))
