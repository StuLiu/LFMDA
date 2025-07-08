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

class Robotic(BaseData):
    LABEL_MAP = OrderedDict(
        # Unlabeled=-1,
        c1=0,
        c2=1,
        c3=2,
        c4=3,
        c5=4,
        c6=5,
        c7=6,
        c8=7,
        c9=8,
        c10=9,
        c11=10,
        c12=11,
        c13=12,
        c14=13,
        c15=14,
        c16=15,
        c17=16,
        c18=17,
        c19=18,
        c20=19,
        c21=20,
        c22=21,
        c23=22,
        c24=23,
        c25=24,
        c26=25,
        c27=26,
        c28=27,
        c29=28,
        c30=29,
        c31=30,
        c32=31,
        c33=32,
        c34=33,
        c35=34,
        c36=35,
        c37=36,
        c38=37,
        c39=38,
        c40=39,
        c41=40,
        c42=41,
        c43=42,
        c44=43,
        c45=44,
        c46=45,
        c47=46,
    )
    COLOR_MAP = OrderedDict(
        #[80, 175, 76],
        c1=[197, 190, 176],
        c2=[211, 211, 211],
        c3=[169, 169, 169],
        c4=[19, 69, 139],
        c5=[0, 100, 0],
        c6=[0, 140, 255],
        c7=[226, 43, 138],
        c8=[193, 182, 255],
        c9=[0, 64, 192],
        c10=[47, 255, 173],
        c11=[180, 105, 255],
        c12=[34, 139, 34],
        c13=[144, 238, 144],
        c14=[0, 255, 127],
        c15=[0, 69, 255],
        c16=[0, 0, 0],
        c17=[128, 128, 128],
        c18=[255, 144, 30],
        c19=[255, 255, 255],
        c20=[0, 0, 255],
        c21=[87, 139, 46],
        c22=[47, 107, 85],
        c23=[47, 255, 173],
        c24=[50, 205, 154],
        c25=[50, 205, 0],
        c26=[130, 0, 75],
        c27=[250, 230, 230],
        c28=[107, 183, 189],
        c29=[169, 169, 169],
        c30=[235, 206, 135],
        c31=[211, 211, 211],
        c32=[105, 105, 105],
        c33=[238, 238, 175],
        c34=[192, 192, 192],
        c35=[0, 215, 255],
        c36=[180, 105, 255],
        c37=[80, 127, 255],
        c38=[0, 100, 0],
        c39=[50, 205, 50],
        c40=[45, 82, 160],
        c41=[152, 251, 152],
        c42=[122, 160, 255],
        c43=[211, 211, 211],
        c44=[0, 215, 255],
        c45=[244, 164, 96],
        c46=[65, 105, 225],
        c47=[240, 230, 140]
    )
    PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    SIZE = (512, 512)
    IGNORE_LABEL = -1

    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id', read_sup=False):
        super().__init__(image_dir, mask_dir, transforms, label_type=label_type,
                         offset=-1, ignore_label=self.IGNORE_LABEL, num_class=len(self.LABEL_MAP),
                         read_sup=read_sup, postfix_image='.png', postfix_label='.png')


if __name__ == '__main__':
    dataset_ = Robotic(image_dir='/home/liuwang/liuwang_data/documents/projects/NonDA2/data/robotic/img_dir/train',
                          mask_dir='/home/liuwang/liuwang_data/documents/projects/NonDA2/data/robotic/img_dir/train')
    print(f'data size = {len(dataset_)}')
    for idx in range(len(dataset_)):
        ret = dataset_[idx]
        print(ret)
