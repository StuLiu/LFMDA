"""
@Project : DAFormer-master
@File    : clip_cosine_similarity.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/27 下午11:03
@e-mail  : 1183862787@qq.com
"""

import torch
import clip
import open_clip
from PIL import Image


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


class ClipCosineSimilarity:

    def __init__(self, backbone="ViT-B/32", device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load(backbone, device=device)

    def encode(self, img_path):
        img_tensor = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_tensor)
        return img_feat

    def __call__(self, tensors1, tensors2):
        simi_mat = torch.cosine_similarity(tensors1, tensors2, dim=1)
        return simi_mat


# class ClassVecCosineSimilarity:
#
#     def __init__(self, backbone="ViT-B/32"):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model, self.preprocess = clip.load(backbone, device=device)
#
#     def encode(self, img_path):
#         img_tensor = self.preprocess(Image.open(img_path)).unsqueeze(0).cuda()
#         img_feat = self.model.encode_image(img_tensor)
#         return img_feat
#
#     def __call__(self, tensors1, tensors2):
#         simi_mat = torch.cosine_similarity(tensors1, tensors2, dim=1)
#         return simi_mat
