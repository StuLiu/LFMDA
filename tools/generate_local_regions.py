"""
@Project : gstda
@File    : local_region_homog.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/18 下午4:58
@e-mail  : 1183862787@qq.com
"""
import warnings
from lfmda.utils.local_region_homog import SAM, get_all_regs


warnings.filterwarnings('ignore')


def demo():
    sam_model = SAM(sam_checkpoint="ckpts/sam_vit_b_01ec64.pth", model_type="vit_b")
    sam_model.get_local_regions(
        image_path='data/LoveDA/Train/Urban/images_png/1366.png',
        save=False,
        show=True
    )


if __name__ == '__main__':
    get_all_regs(img_dir_tgt='data/IsprsDA/Potsdam/img_dir/train')
    get_all_regs(img_dir_tgt='data/IsprsDA/Vaihingen/img_dir/train')
