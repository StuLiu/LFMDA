from configs.ToPotsdam import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS, DATA_ROOT_TGT
import lfmda.aug.augmentation as mag

# teacher cfg
MODEL_TEACHER = 'EvaSeg'
BACKBONE_TEACHER = 'eva02_large_patch14_224'
PRETRAINED_TEACHER = 'ckpts/eva02_large_patch14_224.mim_m38m.bin'
# student cfg
MODEL_STUDENT = 'Deeplabv2'
BACKBONE_STUDENT = 'resnet101'
PRETRAINED_STUDENT = None

BATCH_SIZE = 8

IGNORE_LABEL = -1
MOMENTUM = 0.9

SNAPSHOT_DIR = f'./log/lfmda/{MODEL_STUDENT}_{BACKBONE_STUDENT}/2potsdam_{MODEL_TEACHER}_{BACKBONE_TEACHER}'

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
KD_INTERVAL = 3
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