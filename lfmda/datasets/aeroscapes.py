"""
@Project :
@File    :
@IDE     : PyCharm
@Author  : Wang Liu
@Date    :
@e-mail  : 1183862787@qq.com
"""
import logging
import numpy as np
from lfmda.datasets.basedata import BaseData
from collections import OrderedDict

logger = logging.getLogger(__name__)


# class ids in label image
# Person[192, 128, 128] - -------------1
# Bike[0, 128, 0] - -------------------2
# Car[128, 128, 128] - ----------------3
# Drone[128, 0, 0] - ------------------4
# Boat[0, 0, 128] - -------------------5
# Animal[192, 0, 128] - ---------------6
# Obstacle[192, 0, 0] - ---------------7
# Construction[192, 128, 0] - ---------8
# Vegetation[0, 64, 0] - --------------9
# Road[128, 128, 0] - -----------------10
# Sky[0, 128, 128] - ------------------11

class Aeroscapes(BaseData):
    LABEL_MAP = OrderedDict(
        # Unlabeled=-1,
        Person=0,
        Bike=1,
        Car=2,
        Drone=3,
        Boat=4,
        Animal=5,
        Obstacle=6,
        Construction=7,
        Vegetation=8,
        Road=9,
        Sky=10,
    )
    COLOR_MAP = OrderedDict(
        # Unlabeled=[0, 0, 0],
        Person=[192, 128, 128],
        Bike=[0, 128, 0],
        Car=[128, 128, 128],
        Drone=[128, 0, 0],
        Boat=[0, 0, 128],
        Animal=[192, 0, 128],
        Obstacle=[192, 0, 0],
        Construction=[192, 128, 0],
        Vegetation=[0, 64, 0],
        Road=[128, 128, 0],
        Sky=[0, 128, 128],
    )
    PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    SIZE = (640, 480)
    IGNORE_LABEL = -1

    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id', read_sup=False):
        super().__init__(image_dir, mask_dir, transforms, label_type=label_type,
                         offset=-1, ignore_label=self.IGNORE_LABEL, num_class=len(self.LABEL_MAP),
                         read_sup=read_sup, postfix_image='.jpg', postfix_label='.png')


if __name__ == '__main__':
    dataset_ = Aeroscapes(image_dir='G:\\datasets\\aeroscapes\\JPEGImages',
                          mask_dir='G:\\datasets\\aeroscapes\\SegmentationClass')
    print(f'data size = {len(dataset_)}')
    for idx in range(len(dataset_)):
        ret = dataset_[idx]
        print(ret)
