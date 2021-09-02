#!/usr/bin/python3

import os
import sys
import argparse
import warnings

import torch

warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from BaseSeg.config.config import get_cfg_defaults
from BaseSeg.engine.segmentor_predict import SegmentationInfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='full functional execute script of flare seg module.')
    parser.add_argument('-c', '--config', type=str, default='./predict_config.yaml', help='config file path')
    parser.add_argument('-i', '--input_path', type=str, default='/workspace/inputs/', help='input path')
    parser.add_argument('-o', '--output_path', type=str, default='/workspace/outputs/', help='output path')
    parser.add_argument('-s', '--series_uid_path', type=str, default='', help='series uid')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.DATA_LOADER.TEST_IMAGE_DIR = args.input_path
    cfg.TESTING.SAVER_DIR = args.output_path
    if args.series_uid_path is not None and os.path.exists(args.series_uid_path):
        cfg.DATA_LOADER.TEST_SERIES_IDS_TXT = args.series_uid_path

    if cfg.ENVIRONMENT.CUDA and torch.cuda.is_available():
        cfg.ENVIRONMENT.CUDA = True
    else:
        cfg.ENVIRONMENT.CUDA = False

    cfg.freeze()

    segmentation = SegmentationInfer(cfg)
    segmentation.run()
