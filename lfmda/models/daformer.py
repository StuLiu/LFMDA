import torch
from torch import Tensor
from torch.nn import functional as F
from lfmda.models.base import BaseModel
from lfmda.models.heads.daformer import DAFormerHead


class DAFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = DAFormerHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y0 = self.backbone(x[:, :3, :, :])
        y = self.decode_head(y0)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)  # to original image shape
        if self.training:
            return y, y, y0[-1]
        return y.softmax(dim=1)


if __name__ == '__main__':
    model = DAFormer('MiT-B5')
    model.init_pretrained('../../ckpts/backbones/mit/mit_b5.pth')
    _x = torch.zeros(4, 3, 512, 512)
    _y = model(_x)
    print(_y[0].shape, _y[1].shape, _y[-1].shape)
