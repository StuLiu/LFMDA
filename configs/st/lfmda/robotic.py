from albumentations import *
import ever as er
from configs.robotic import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS
import lfmda.aug.augmentation as mag

MODEL = 'SegFormer'
BACKBONE = 'MiT-B2'
PRETRAINED = 'ckpts/backbones/mit/mit_b2.pth'

MODEL_STUDENT = 'SegFormer'
BACKBONE_STUDENT = 'MiT-B2'
PRETRAINED_STUDENT = 'ckpts/backbones/mit/mit_b2.pth'

IGNORE_LABEL = -1
MOMENTUM = 0.9

SNAPSHOT_DIR = f'./log/lfmda/{MODEL}_{BACKBONE}/robotic_src'

# Hyper Paramters
OPTIMIZER = 'AdamW'
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
NUM_STEPS = 10000
STAGE1_STEPS = 10000
STAGE2_STEPS = 10000
STAGE3_STEPS = 10000
POWER = 0.9                 # lr poly power
EVAL_EVERY = 1000
GENE_EVERY = 1000
CUTOFF_TOP = 0.9
CUTOFF_LOW = 0.9
