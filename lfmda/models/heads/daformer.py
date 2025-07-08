import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
from lfmda.models.heads.segformer import MLP


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv_last = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_last = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu(self.bn_last(self.conv_last(x)))
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class DAFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.aspp_fuse = ASPP(embed_dim * 4, embed_dim)
        # self.aspp_fuse = ConvModule(embed_dim*4, embed_dim)
        # self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.aspp_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg


if __name__ == '__main__':
    x = torch.randn(4, 512, 128, 128)
    aspp = ASPP(512, 256)
    y = aspp(x)
    print(y.shape)
    head_ = DAFormerHead([64,128,256,512], 256, 6)
    x1 = torch.randn(4, 64, 128, 128)
    x2 = torch.randn(4, 128, 64, 64)
    x3 = torch.randn(4, 256, 32, 32)
    x4 = torch.randn(4, 512, 16, 16)
    y = head_((x1, x2, x3, x4))
    print(y.shape)