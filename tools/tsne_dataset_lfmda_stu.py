"""
@Project : rads2
@File    : tsne2.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/4/30 下午5:32
@e-mail  : 1183862787@qq.com
"""
import logging
import argparse
import os
import random

import cv2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from numpy import reshape
from keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.manifold import TSNE
from ever.core.iterator import Iterator

from lfmda.aug.augmentation import Normalize
from lfmda.datasets.daLoader import DALoader
from lfmda.utils.tools import *
from lfmda.datasets import *
from lfmda.models import get_model
from lfmda.models.Encoder import Deeplabv2
from lfmda.gast.alignment import DownscaleLabel


def hello_world():
    def get_data():
        digits = datasets.load_digits(n_class=6)
        data = digits.data
        label = digits.target
        n_samples, n_features = data.shape
        return data, label, n_samples, n_features

    x, y, n, k = get_data()

    # iris = load_iris()
    # x = iris.data
    # y = iris.target

    print(f'input x shape={x.shape}')
    print(f'input y shape={y.shape}')

    tsne = TSNE(n_components=2, verbose=1, random_state=123, n_iter=1000)
    z = tsne.fit_transform(x)
    print(f'output z shape={z.shape}')

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1",
                    y="comp-2",
                    hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 6),
                    data=df).set(title="Iris data T-SNE projection")
    plt.show()


# hello_world()


class TSNECrossDomain:

    def __init__(self, n_components=2, verbose=1, random_state=123, n_iter=1000,
                 domains=('Vaihingen', 'Potsdam'), show_class=None,
                 n_class=7, ignore_label=-1, logger=None):
        self.tsne = TSNE(n_components=n_components,
                         verbose=verbose,
                         random_state=random_state,
                         n_iter=n_iter)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
        self.domains = domains
        self.class_num = n_class
        self.ignore_label = ignore_label
        self.eps = 1e-7
        self.downscale = DownscaleLabel(scale_factor=16, n_classes=n_class, ignore_label=-1, min_ratio=0.75)
        self.show_class = tuple(range(1, 5)) if not show_class else show_class
        # self.show_class = tuple(range(0, 6)) if not show_class else show_class
        self.normalzie = Normalize(
            mean=(73.53223948, 80.01710095, 74.59297778),
            std=(41.5113661, 35.66528876, 33.75830885)
        )

    def _index2onehot(self, label):
        """Compute the one-hot label
        Args:
            label: torch.Tensor, gt or pseudo label, shape=(b, 1, h, w)
        Returns:
            labels: (b*h*w, c)
        """
        labels = label.clone()
        if len(label.shape) < 4:
            labels = labels.unsqueeze(dim=1)
        labels = labels.permute(0, 2, 3, 1).contiguous().view(-1, 1)  # (b*h*w, 1)
        labels[labels == self.ignore_label] = self.class_num
        labels = tnf.one_hot(labels.squeeze(1), num_classes=self.class_num + 1)[:, :-1]  # (b*h*w, c)
        return labels.contiguous()

    def _compute_local_prototypes(self, feat, label):
        """Compute prototypes within a mini-batch
        Args:
            feat: torch.Tensor, mini-batch features, shape=(b, k, h, w)
            label: torch.Tensor, label(the gt or pseudo label, instead of logits), shape=(b, 1, h, w)
            update: bool, if update the global prototypes
            decay: float in (0, 1), the parameter for ema algorithm. the higher, update slower.

        Returns:
            local_prototype: class prototypes within a mini-batch. shape=(c, k)
            n_instance: pixel number for each class.
        """
        b, k, h, w = feat.shape
        feats = feat.permute(0, 2, 3, 1).contiguous().view(-1, k)  # (b*h*w, k)
        feats = feats.unsqueeze(dim=1)  # (b*h*w, 1, k)

        labels = self._index2onehot(label)  # (b, 1, h, w) -> (b*h*w, c)
        labels = labels.unsqueeze(dim=-1)  # (b*h*w, c, 1)

        n_instance = labels.sum(0).expand(self.class_num, k)  # (c, k)
        local_prototype = (feats * labels).sum(0) / (n_instance + self.eps)  # (c, k)

        return local_prototype, n_instance[:, 0]

    def _plot_and_save(self, df, save_path):
        ax = sns.scatterplot(x="comp-1",
                             y="comp-2",
                             hue=df.y.tolist(),
                             style="domain",
                             legend=False,
                             palette=sns.color_palette("hls", 6),
                             data=df)
        ax.set(title="")
        fig = ax.get_figure()
        fig.savefig(save_path, dpi=400)
        plt.show()

    @torch.no_grad()
    def random_choose_from_datasets(self, model, src_loader, tgt_loader, n_images=10, n_pixels=50, save_path='tsne.png'):
        model.train()
        df = pd.DataFrame()
        feat_all, labels_all, domain_flags = [], [], []
        for _ in tqdm(range(n_images)):
            # source infer
            batch = src_loader.next()
            images_s, labels_s = batch[0]
            images_s, labels_s = images_s.cuda(), labels_s['cls'].cuda()
            _, _, feat_s = model(images_s)
            b, k = feat_s.shape[:2]
            labels_s = self.downscale(labels_s).contiguous()
            feat_s = tnf.interpolate(feat_s, size=labels_s.shape[-2:], mode='bilinear')
            labels_s = labels_s.view(-1, )
            feat_s = feat_s.permute(0, 2, 3, 1).contiguous().view(-1, k).detach()

            for _ in range(n_pixels):
                idx = random.randrange(0, feat_s.shape[0])
                if labels_s[idx] in self.show_class:
                    feat_all.append(feat_s[idx, :])
                    labels_all.append(labels_s[idx: idx + 1])
                    domain_flags.append(self.domains[0])

            batch = tgt_loader.next()
            images_t, labels_t = batch[0]
            images_t, labels_t = images_t.cuda(), labels_t['cls'].cuda()
            _, _, feat_t = model(images_t)
            b, k = feat_t.shape[:2]
            labels_t = self.downscale(labels_t).contiguous()
            feat_t = tnf.interpolate(feat_t, size=labels_t.shape[-2:], mode='bilinear')
            labels_t = labels_t.view(-1, )
            feat_t = feat_t.permute(0, 2, 3, 1).contiguous().view(-1, k).detach()

            for _ in range(n_pixels):
                idx = random.randrange(0, feat_t.shape[0])
                if labels_t[idx] in self.show_class:
                    feat_all.append(feat_t[idx, :])
                    labels_all.append(labels_t[idx: idx + 1])
                    domain_flags.append(self.domains[1])

        feat_all = torch.stack(feat_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        z = self.tsne.fit_transform(feat_all.cpu().numpy())

        df["y"] = labels_all.cpu().numpy()
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        df["domain"] = domain_flags

        self._plot_and_save(df, save_path)

    @torch.no_grad()
    def plot_an_image(self, img_path_src, label_path_src,
                      img_path_tgt, label_path_tgt,
                      model, max_pixel=200, save_dir='.', da=False):

        img_src = cv2.cvtColor(cv2.imread(img_path_src, flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img_src = torch.from_numpy(img_src).unsqueeze(dim=0).permute(0, 3, 1, 2).cuda().float()
        img_tgt = cv2.cvtColor(cv2.imread(img_path_tgt, flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img_tgt = torch.from_numpy(img_tgt).unsqueeze(dim=0).permute(0, 3, 1, 2).cuda().float()
        img = torch.cat([img_src, img_tgt], dim=0)

        lbl_src = cv2.imread(label_path_src, flags=cv2.IMREAD_UNCHANGED)
        lbl_src = torch.from_numpy(lbl_src).unsqueeze(dim=0).cuda().long()
        lbl_tgt = cv2.imread(label_path_tgt, flags=cv2.IMREAD_UNCHANGED)
        lbl_tgt = torch.from_numpy(lbl_tgt).unsqueeze(dim=0).cuda().long()
        lbl = torch.cat([lbl_src, lbl_tgt], dim=0)

        img, lbl, _ = self.normalzie(img, lbl, None)
        print(img.shape, lbl.shape)

        lbl_down = self.downscale(lbl).contiguous().view(2, -1, )

        df = pd.DataFrame()
        feat_all, labels_all, domain_flags = [], [], []

        model.train()
        _, _, feats = model(img)

        _, k, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).contiguous().view(2, -1, k).detach()
        for domain_idx in [0, 1]:
            for idx in range(0, h * w):
                if lbl_down[domain_idx, idx: idx + 1] in self.show_class:
                    feat_all.append(feats[domain_idx, idx, :])
                    labels_all.append(lbl_down[domain_idx, idx: idx + 1])
                    domain_flags.append(self.domains[domain_idx])

        feat_all = torch.stack(feat_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        z = self.tsne.fit_transform(feat_all.cpu().numpy())

        df["y"] = labels_all.cpu().numpy()
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        df["domain"] = domain_flags

        self._plot_and_save(df, os.path.join(save_dir, f"compare_img_{'da' if da else 'src'}.png"))


if __name__ == '__main__':
    # hello_world()
    seed_torch(2333)

    parser = argparse.ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str,
                        default='st.lfmda.2potsdam_deeplabv2_tsne', help='config path')
    parser.add_argument('--ckpt', type=str,
                        default='log/lfmda/Deeplabv2_resnet101/2potsdam/src_warmup_st-pcd/Potsdam_stu_curr.pth')
    # parser.add_argument('--ckpt', type=str,
    #                     default='log/lfmda/Deeplabv2_resnet101/2potsdam/ssl_proto_sam_0.95_68.78/Potsdam_stu_curr.pth')
    args = parser.parse_args()

    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = 'log/tsne'
    os.makedirs(log_dir, exist_ok=True)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='TSNE', logdir=log_dir)
    save_path_stu = os.path.join(log_dir, f'random_choose_from_datasets_pcd.png')

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name_stu = str(cfg.MODEL_STUDENT)
    backbone_stu = str(cfg.BACKBONE_STUDENT)
    pretrained_stu = str(cfg.PRETRAINED_STUDENT)

    # model for semantic segmentation
    model_stu, feat_channel, down_scale = get_model(class_num, model_name_stu, backbone_stu, pretrained_stu)
    model_stu = model_stu.cuda()
    model_stu.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))

    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)

    tsne_ = TSNECrossDomain(logger=logger, n_class=class_num, show_class=(1,2,3,4))
    tsne_.random_choose_from_datasets(model_stu, sourceloader_iter, targetloader_iter,
                                      n_images=20, save_path=save_path_stu)
