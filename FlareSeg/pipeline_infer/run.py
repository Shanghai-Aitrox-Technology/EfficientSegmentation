#!/usr/bin/python3

"""
pipeline of flare segmentation.
"""

import os
import sys
import argparse
import warnings

import torch

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.gpu_utils import set_gpu
from BaseSeg.config.config import get_cfg_defaults
from BaseSeg.engine.segmentor import BaseSegmentation3D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='full functional execute script of fine flare seg module.')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('-c', '--config', type=str, default='./config.yaml', help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.ENVIRONMENT.RANK = str(args.local_rank)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg.ENVIRONMENT.DATA_BASE_DIR = os.path.join(base_dir, cfg.ENVIRONMENT.DATA_BASE_DIR)

    set_gpu(cfg.ENVIRONMENT.NUM_GPU, used_percent=0.2, local_rank=args.local_rank)
    if cfg.ENVIRONMENT.CUDA and torch.cuda.is_available():
        cfg.ENVIRONMENT.CUDA = True
        torch.cuda.manual_seed_all(cfg.ENVIRONMENT.SEED)
    else:
        cfg.ENVIRONMENT.CUDA = False
        torch.manual_seed(cfg.ENVIRONMENT.SEED)

    if cfg.TRAINING.IS_DISTRIBUTED_TRAIN and cfg.ENVIRONMENT.NUM_GPU > 1:
        cfg.TRAINING.IS_DISTRIBUTED_TRAIN = True
    else:
        cfg.TRAINING.IS_DISTRIBUTED_TRAIN = False

    if cfg.ENVIRONMENT.DATA_BASE_DIR is not None:
        cfg.DATA_LOADER.TRAIN_DB_FILE = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_LOADER.TRAIN_DB_FILE
        cfg.DATA_LOADER.VAL_DB_FILE = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_LOADER.VAL_DB_FILE
        cfg.DATA_LOADER.TEST_DB_FILE = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_LOADER.TEST_DB_FILE

    cfg.freeze()

    detection = BaseSegmentation3D(cfg, phase='test')
    detection.do_test()
