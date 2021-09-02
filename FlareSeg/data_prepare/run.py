#!/usr/bin/python3

import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from BaseSeg.config.config import get_cfg_defaults
from BaseSeg.data.data_prepare import run_prepare_data


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./config.yaml')
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg.ENVIRONMENT.DATA_BASE_DIR = os.path.join(base_dir, cfg.ENVIRONMENT.DATA_BASE_DIR)
    if cfg.DATA_PREPARE.OUT_DIR is None:
        cfg.DATA_PREPARE.OUT_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR
    if cfg.ENVIRONMENT.DATA_BASE_DIR is not None:
        cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT
        cfg.DATA_PREPARE.TEST_SERIES_IDS_TXT = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.TEST_SERIES_IDS_TXT
        cfg.DATA_PREPARE.TRAIN_IMAGE_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.TRAIN_IMAGE_DIR
        cfg.DATA_PREPARE.TRAIN_MASK_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.TRAIN_MASK_DIR
        cfg.DATA_PREPARE.TEST_IMAGE_DIR = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.TEST_IMAGE_DIR
        cfg.DATA_PREPARE.DEFAULT_TRAIN_DB = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.DEFAULT_TRAIN_DB
        cfg.DATA_PREPARE.DEFAULT_VAL_DB = cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_PREPARE.DEFAULT_VAL_DB
    cfg.freeze()

    run_prepare_data(cfg)
