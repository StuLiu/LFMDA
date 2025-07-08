'''
@Project : NonDA2 
@File    : clip_seg.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/11 下午2:39
@e-mail  : 1183862787@qq.com
'''

import clip
import open_clip
from PIL import Image

import math
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from collections import OrderedDict


remoteclip_seg_settings = {
    'ViT-B-32': {
        'width': 768,
        'layers': 12,
    },
    'ViT-L-14': {
        'width': 1024,
        'layers': 24,
    },
}

class RemoteClipCosineSimilarity:

    def __init__(self, backbone="ViT-B-32", ckpt_path='ckpts/RemoteCLIP-ViT-B-32.pt', device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(backbone)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        message = self.model.load_state_dict(ckpt)
        print(message)
        self.model = self.model.to(self.device).eval()

    def encode(self, img_path):
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_tensor = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img_tensor)
            return img_feat

    def __call__(self, tensors1, tensors2):
        simi_mat = torch.cosine_similarity(tensors1, tensors2, dim=1)
        return simi_mat


def get_open_clip_vit(model_name, img_size, patch_size):
    settings = remoteclip_seg_settings[model_name]
    model_kwargs = {
        'vision_cfg':{
            'pool_type': 'none',
            'image_size': img_size,
            'patch_size': patch_size, **settings
        }
    }
    model_clip, _, _ = open_clip.create_model_and_transforms(model_name, **model_kwargs)
    encoder = model_clip.visual
    if hasattr(encoder, "proj"):
        encoder.proj = None
    return encoder


class RemoteClipSeg(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int = None,
            patch_size: int = 14,
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

        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(backbone)

        self.encoder = get_open_clip_vit(model_name, self.vit_in_img_size, patch_size)
        self.backbone = self.encoder
        self.embed_dim = remoteclip_seg_settings[model_name]['width']
        self.num_concat_last_layers = 4
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim,
                               self.embed_dim // 2,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim // 2,
                               self.embed_dim // 4,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.embed_dim // 4),
            nn.Conv2d(
                self.embed_dim // 4,
                self.embed_dim // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.embed_dim // 4),
        )

        out = nn.Conv2d(self.embed_dim // 4, num_classes, kernel_size=1, padding=0, bias=False)
        torch.nn.init.normal_(out.weight, 0, std=0.1)
        self.out = out

        self.param_defs_decoder = [
            ("out", self.out),
            ("upscale", self.upscale)
        ]
        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.transformer.resblocks),
        ]
        self.encoder_depth = len(self.encoder.transformer.resblocks)

    def forward_features(self, img: torch.Tensor):
        b, c, h, w = img.shape
        token_img_shape = (b, self.embed_dim, h // self.patch_size, w // self.patch_size)
        x = self.encoder(img)
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
                if str(k).startswith('visual.'):
                    state_dict_new[str(k).replace('visual.', '')] = v
            if 'positional_embedding' in state_dict_new.keys():
                state_dict_new.pop('positional_embedding')
            state_dict_new.pop('proj')
            self.encoder.load_state_dict(state_dict_new, strict=strict)

    # def init_pretrained2(self, pretrained: str = None, strict=True) -> None:
    #     if pretrained or pretrained == 'None':
    #         self.encoder.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=strict)


if __name__ == '__main__':
    _x = torch.zeros(2, 3, 512, 512).cuda()

    vit = RemoteClipSeg(512, 'ViT-L-14', num_classes=6, in_img_scale=1.1, patch_size=14).cuda()

    vit.init_pretrained('ckpts/backbones/vit/RemoteCLIP-ViT-L-14.pt')
    y_ = vit(_x)
    print(y_[0].shape, y_[-1].shape)
