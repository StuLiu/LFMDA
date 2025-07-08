"""
@Project : rads2
@File    : __init__.py.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/5/26 下午4:53
@e-mail  : 1183862787@qq.com
"""

import logging
from lfmda.models.Discriminator import FCDiscriminator
from lfmda.models.Encoder import Deeplabv2
from lfmda.models.segformer import SegFormer
from lfmda.models.daformer import DAFormer
from lfmda.models.backbones import *
from lfmda.models.vitseg import VitSeg, vitseg_settings
from lfmda.models.eva_seg import EvaSeg, eva_seg_settings
from lfmda.models.mae_seg import MAESeg, mae_seg_settings
from lfmda.models.clip_seg import RemoteClipSeg, remoteclip_seg_settings


model_names = ['Deeplabv2', 'SegFormer', 'DAFormer', 'VitSeg', 'EvaSeg', 'MAESeg', 'RemoteClipSeg']


def get_model(class_num, model_name, backbone_name: str = 'resnet101', pretrained: str = None):
    logging.info(f'model_name={model_name}, backbone_name={backbone_name}, pretrained={pretrained}')
    assert model_name in model_names, f'model {model_name} is not in {model_names}'
    down_scale = 32
    if model_name == 'Deeplabv2':
        feat_channel = 1024
        model = Deeplabv2(dict(
            backbone=dict(
                resnet_type=backbone_name,
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=True,
            cascade=False,
            use_ppm=True,
            ppm=dict(
                num_classes=class_num,
                use_aux=False,
                fc_dim=feat_channel,
            ),
            inchannels=feat_channel,
            num_classes=class_num,
            is_ins_norm=True,
        ))
        down_scale = 16
    elif model_name in ['SegFormer', 'DAFormer']:
        mit_version = backbone_name.split('-')[-1]
        feat_channel = mit_settings[mit_version][0][-1]
        model = eval(model_name)(backbone=backbone_name, num_classes=class_num)
        model.init_pretrained(pretrained)
    elif model_name in ['VitSeg']:
        down_scale = None
        assert backbone_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        feat_channel = vitseg_settings[backbone_name]['feat_channel']
        model = VitSeg(img_size=512, model_name=backbone_name, num_classes=class_num, patch_size=14)
        model.init_pretrained(pretrained)
    elif model_name in ['EvaSeg']:
        down_scale = None
        assert backbone_name in ['eva02_large_patch14_224', 'eva02_large_patch14_clip_224']
        feat_channel = eva_seg_settings[backbone_name]['feat_channel']
        model = EvaSeg(img_size=512, model_name=backbone_name, num_classes=class_num, patch_size=14)
        model.init_pretrained(pretrained)
    elif model_name in ['MAESeg']:
        down_scale = None
        assert backbone_name in ['vit_base_patch16_224', 'vit_large_patch16_224']
        feat_channel = mae_seg_settings[backbone_name]['feat_channel']
        model = MAESeg(img_size=512, model_name=backbone_name, num_classes=class_num, patch_size=16)
        model.init_pretrained(pretrained)
    elif model_name in ['RemoteClipSeg']:
        down_scale = None
        assert backbone_name in ['ViT-L-14', 'ViT-B-32']
        feat_channel = remoteclip_seg_settings[backbone_name]['width']
        model = RemoteClipSeg(img_size=512, model_name=backbone_name, num_classes=class_num, patch_size=14)
        model.init_pretrained(pretrained)
    else:
        raise NotImplemented
    return model, feat_channel, down_scale


__all__ = ['get_model', 'FCDiscriminator'] + model_names


if __name__ == '__main__':
    get_model(6, 'VitSeg', 'dinov2_vitb14')
