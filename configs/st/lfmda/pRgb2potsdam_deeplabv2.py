from configs.ToPotsdam import EVAL_DATA_CONFIG, PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, \
    TARGET_SET, target_dir, DATASETS, DATA_ROOT_TGT
import lfmda.aug.augmentation as mag
import albumentations as alb
import ever as er


source_dir = dict(
    image_dir=[
        'data/IsprsDA/Potsdam_rgb/img_dir/train',
    ],
    mask_dir=[
        'data/IsprsDA/Potsdam_rgb/ann_dir/train',
    ],
)

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=alb.Compose([
        alb.RandomCrop(512, 512),
        alb.OneOf([
            alb.HorizontalFlip(True),
            alb.VerticalFlip(True),
            alb.RandomRotate90(True)
        ], p=0.75),
        alb.Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=4,
)

# teacher cfg
MODEL_TEACHER = 'VitSeg'
BACKBONE_TEACHER = 'dinov2_vitl14'
PRETRAINED_TEACHER = 'ckpts/dinov2_vitl14_pretrain.pth'
# student cfg
MODEL_STUDENT = 'Deeplabv2'
BACKBONE_STUDENT = 'resnet101'
PRETRAINED_STUDENT = None

BATCH_SIZE = 8

IGNORE_LABEL = -1
MOMENTUM = 0.9

SNAPSHOT_DIR = f'./log/lfmda/{MODEL_STUDENT}_{BACKBONE_STUDENT}/pRgb2potsdam'

# Hyper Paramters
OPTIMIZER = 'AdamW'
WEIGHT_DECAY = 0.01
LEARNING_RATE = 6e-5
# WEIGHT_DECAY = 0.0005
# LEARNING_RATE = 1e-2

STAGE1_STEPS = 2000
STAGE2_STEPS = 6000
NUM_STEPS = None        # for learning rate poly
PREHEAT_STEPS = 500    # for warm-up
POWER = 0.9             # lr poly power
EVAL_EVERY = 1000
GENE_EVERY = 1000
KD_INTERVAL = 1
KD_TEM = 0.3
CUTOFF_TOP = 0.8
CUTOFF_LOW = 0.6

TARGET_DATA_CONFIG = dict(
    data_root=DATA_ROOT_TGT,
    image_dir=target_dir['image_dir'],
    mask_dir=[None],
    transforms=mag.Compose([
        mag.RandomCrop((512, 512)),
        mag.RandomHorizontalFlip(0.5),
        mag.RandomVerticalFlip(0.5),
        mag.RandomRotate90(0.5),
        mag.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            clamp=True,
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    label_type='prob',
    read_sup=True,
)