"""
@Project : DAFormer-master
@File    : cross_domain_similarity.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/27 下午5:38
@e-mail  : 1183862787@qq.com
"""
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./cat.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.002611 0.00804  0.9893  ]]


image_cat = preprocess(Image.open("./cat.png")).unsqueeze(0).to(device)
image_dog1 = preprocess(Image.open("./dog1.png")).unsqueeze(0).to(device)
image_dog2 = preprocess(Image.open("./dog2.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_cat_features = model.encode_image(image_cat)
    image_dog1_features = model.encode_image(image_dog1)
    image_dog2_features = model.encode_image(image_dog2)

    simi_mat = torch.cosine_similarity(
        image_dog1_features,
        torch.cat([image_cat_features, image_dog2_features], dim=0),
        dim=1
    )

    print("simi_mat:", simi_mat)  # prints: [0.6875, 0.6968]
