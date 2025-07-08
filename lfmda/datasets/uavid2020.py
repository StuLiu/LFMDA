"""
@Project :
@File    :
@IDE     : PyCharm
@Author  : Wang Liu
@Date    :
@e-mail  : 1183862787@qq.com
"""
import numpy as np
from lfmda.datasets.basedata import BaseData
from collections import OrderedDict

"""UAVid2020 dataset.

In segmentation map annotation for UAVid2020, 0 stands for background, which is
included in 8 categories. ``reduce_zero_label`` is fixed to False. The
``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png', too.
In UAVid2020, 200 images for training, 70 images for validating, and 150 images for testing.
The 8 classes and corresponding label color (R,G,B) are as follows:
    'label name'        'R,G,B'         'label id'
    Background clutter  (0,0,0)         0
    Building            (128,0,0)       1
    Road                (128,64,128)    2
    Static car          (192,0,192)     3
    Tree                (0,128,0)       4
    Low vegetation      (128,128,0)     5
    Human               (64,64,0)       6
    Moving car          (64,0,128)      7

"""


class UAVid2020(BaseData):
    LABEL_MAP = OrderedDict(
        Background=0,
        Building=1,
        Road=2,
        Static_car=3,
        Tree=4,
        Low_vegetation=5,
        Human=6,
        Moving_car=7,
    )
    COLOR_MAP = OrderedDict(
        Background=[0, 0, 0],
        Building=[128, 0, 0],
        Road=[128, 64, 128],
        Static_car=[192, 0, 192],
        Tree=[0, 128, 0],
        Low_vegetation=[128, 128, 0],
        # Low_vegetation=[0, 128, 0],
        # Human=[64, 64, 0],
        Human=[68, 114, 196],
        Moving_car=[192, 0, 192],
    )
    PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    SIZE = (1080, 1920)
    IGNORE_LABEL = -1

    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id', read_sup=False):
        super().__init__(image_dir, mask_dir, transforms, label_type=label_type,
                         offset=0, ignore_label=self.IGNORE_LABEL, num_class=len(self.LABEL_MAP),
                         read_sup=read_sup, postfix_image='.png', postfix_label='.png')


if __name__ == '__main__':
    dataset_ = UAVid2020(image_dir='/home/liuwang/liuwang_data/documents/projects/NonDA2/data/uavid2020/img_dir/train',
                         mask_dir='/home/liuwang/liuwang_data/documents/projects/NonDA2/data/uavid2020/ann_dir/train')
    print(f'data size = {len(dataset_)}')
    import cv2
    from lfmda.utils.tools import overlay_segmentation_cv2

    for idx in range(len(dataset_)):
        ret = dataset_[idx]
        img = ret[0]
        lbl = ret[1]['cls']
        print(img.shape, lbl.shape, np.unique(lbl))
        lbl = lbl[:, :, 0]
        vis_rgb = overlay_segmentation_cv2(img.squeeze(), lbl.squeeze(), UAVid2020.COLOR_MAP.values(), alpha=0.68)
        cv2.imshow('', cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
