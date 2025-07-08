'''
@Project : NonDA2 
@File    : mae_seg.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/11 下午10:58
@e-mail  : 1183862787@qq.com
'''
# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import math
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from collections import OrderedDict


mae_seg_settings = {
    'vit_base_patch16_224': {
        'feat_channel': 768
    },
    'vit_large_patch16_224': {
        'feat_channel': 1024
    },
}


def get_timm_vit(name: str, img_size: int, patch_size: int):
    encoder = timm.create_model(
        name,
        pretrained=False,
        img_size=img_size,
        patch_size=patch_size,
    )

    if hasattr(encoder, "fc_norm"):
        encoder.fc_norm = nn.Identity()
    if hasattr(encoder, "neck"):
        encoder.neck = nn.Identity()
    if hasattr(encoder, "head"):
        encoder.head = nn.Identity()
    return encoder


class MAESeg(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int = None,
            patch_size: int = 16,
            in_img_scale: float = 1.1,
            align_corners: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.align_corners = align_corners

        if 1.0 == in_img_scale:
            self.vit_in_img_size = int(math.ceil(float(img_size) / self.patch_size) * self.patch_size)
        else:
            self.vit_in_img_size = int((img_size * in_img_scale) // self.patch_size * self.patch_size)

        self.encoder = get_timm_vit(model_name, self.vit_in_img_size, patch_size)
        self.backbone = self.encoder

        self.num_concat_last_layers = 4
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(self.encoder.embed_dim,
                               self.encoder.embed_dim // 2,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.encoder.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.encoder.embed_dim // 2,
                               self.encoder.embed_dim // 4,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.encoder.embed_dim // 4),
            nn.Conv2d(
                self.encoder.embed_dim // 4,
                self.encoder.embed_dim // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.encoder.embed_dim // 4),
        )

        out = nn.Conv2d(self.encoder.embed_dim // 4, num_classes, kernel_size=1, padding=0, bias=False)
        torch.nn.init.normal_(out.weight, 0, std=0.1)
        self.out = out

        self.param_defs_decoder = [
            ("out", self.out),
            ("upscale", self.upscale)
        ]

        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.blocks),
        ]

        self.param_defs_encoder_stems = [
            ("encoder.norm", self.encoder.norm),
            ("encoder.pos_embed", self.encoder.pos_embed),
            ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
            ("encoder.cls_token", self.encoder.cls_token)
            if hasattr(self.encoder, "cls_token")
            else (None, None),
        ]

        self.encoder_depth = len(self.encoder.blocks)

    def forward_features(self, img: torch.Tensor):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)
        x = self.encoder._pos_embed(x_patch)
        x = self.encoder.norm_pre(x)
        x = x.contiguous()
        for i in range(self.encoder_depth):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = self.token_to_image(x, token_img_shape)
        return x

    def forward(self, img):
        b, c, h, w = img.shape
        assert h == w
        orig_img_size = [h, w]
        img = F.interpolate(
            img,
            size=(self.vit_in_img_size, self.vit_in_img_size),
            mode="bilinear", align_corners=self.align_corners
        )
        feats = self.forward_features(img)
        logit = self.out(self.upscale(feats))
        logit = F.interpolate(logit, orig_img_size, mode="bilinear", align_corners=self.align_corners)
        if self.training:
            return logit, logit, feats
        else:
            return logit.softmax(dim=1)

    def token_to_image(self, x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x

    def freeze_backbone(self, freezing=True):
        for param in self.backbone.parameters():
            param.requires_grad = (not freezing)

    def init_pretrained(self, pretrained: str = None, strict=False) -> None:
        if pretrained or pretrained == 'None':
            state_dict = torch.load(pretrained, map_location='cpu')
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                state_dict_new[str(k).replace('visual.trunk.', '')] = v
            if 'pos_embed' in state_dict_new.keys():
                state_dict_new.pop('pos_embed')
            self.encoder.load_state_dict(state_dict_new, strict=strict)

    # def init_pretrained2(self, pretrained: str = None, strict=True) -> None:
    #     if pretrained or pretrained == 'None':
    #         self.encoder.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=strict)


if __name__ == '__main__':
    _x = torch.zeros(2, 3, 512, 512).cuda()

    vit = MAESeg(512, 'timm/vit_large_patch16_224', num_classes=6, patch_size=16).cuda()

    vit.init_pretrained('/home/liuwang/liuwang_data/documents/projects/NonDA2/'
                        'ckpts/backbones/mae/vit_large_patch16_224.mae.pth')
    y_ = vit(_x)
    print(y_[0].shape, y_[-1].shape)