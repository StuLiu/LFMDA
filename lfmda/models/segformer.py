import torch
from torch import Tensor
from torch.nn import functional as F
from lfmda.models.base import BaseModel
from lfmda.models.heads.segformer import SegFormerHead


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels[:-1] + [1024],
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)
        self.backbone.mapper = torch.nn.Conv2d(self.backbone.channels[-1], 1024, 1, 1)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y0 = list(self.backbone(x[:, :3, :, :]))
        y0[-1] = self.backbone.mapper(y0[-1])
        y = self.decode_head(y0)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)  # to original image shape
        if self.training:
            return y, y, y0[-1]
        return y.softmax(dim=1)


if __name__ == '__main__':
    model = SegFormer('MiT-B5')
    model.init_pretrained('../../ckpts/backbones/mit/mit_b5.pth')
    _x = torch.zeros(1, 3, 512, 512)
    _y = model(_x)
    print(_y[0].shape, _y[1].shape, _y[-1].shape)
