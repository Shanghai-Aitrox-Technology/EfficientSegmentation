#!/usr/bin/python3

"""
segmentation of flare in fine resolution.
"""

import os
import sys
import time
import warnings
import argparse

import torch

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from BaseSeg.config.config import get_cfg_defaults
from Common.gpu_utils import set_gpu, run_multiprocessing
from BaseSeg.engine.segmentor_multiprocess import SegmentationMultiProcess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='full functional execute script of fine flare seg module.')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml',
                        help='config file path')
    parser.add_argument('--local_rank', type=str, default='0',
                        help='local rank for multi-gpu training')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.ENVIRONMENT.RANK = args.local_rank
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg.ENVIRONMENT.DATA_BASE_DIR = os.path.join(base_dir, cfg.ENVIRONMENT.DATA_BASE_DIR)
    cfg.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT = os.path.join(base_dir, cfg.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT)

    set_gpu(cfg.ENVIRONMENT.NUM_GPU, used_percent=0.2, local_rank=int(args.local_rank))
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

    train_db_path = cfg.DATA_LOADER.TRAIN_DB_FILE
    val_db_path = cfg.DATA_LOADER.VAL_DB_FILE
    test_db_path = cfg.DATA_LOADER.TEST_DB_FILE

    tune_params = {
        'experiment_1': {'META_ARCHITECTURE': 'EfficientSegNet',
                         'NUM_BLOCKS': [2, 2, 2, 2],
                         'DECODER_NUM_BLOCK': 1,
                         'ENCODER_CONV_BLOCK': 'ResBaseConvBlock',
                         'DECODER_CONV_BLOCK': 'AnisotropicConvBlock',
                         'CONTEXT_BLOCK': 'AnisotropicAvgPooling',
                         'NUM_DEPTH': 4, 'LOSS': 'dice', 'METRIC': 'dice'},
    }

    torch.cuda.synchronize()
    start_time = time.time()
    print('Start training, time: {}'.format(start_time))
    for exp_name, exp_config in tune_params.items():
        print("{} is processing...".format(exp_name))
        cfg.ENVIRONMENT.EXPERIMENT_NAME = exp_name
        cfg.FINE_MODEL.META_ARCHITECTURE = exp_config['META_ARCHITECTURE']

        cfg.FINE_MODEL.NUM_BLOCKS = exp_config['NUM_BLOCKS']
        cfg.FINE_MODEL.DECODER_NUM_BLOCK = exp_config['DECODER_NUM_BLOCK']
        cfg.FINE_MODEL.ENCODER_CONV_BLOCK = exp_config['ENCODER_CONV_BLOCK']
        cfg.FINE_MODEL.DECODER_CONV_BLOCK = exp_config['DECODER_CONV_BLOCK']
        cfg.FINE_MODEL.CONTEXT_BLOCK = exp_config['CONTEXT_BLOCK']
        cfg.FINE_MODEL.NUM_DEPTH = exp_config['NUM_DEPTH']
        cfg.TRAINING.LOSS = exp_config['LOSS']
        cfg.TRAINING.METRIC = exp_config['METRIC']

        # if cfg.FINE_MODEL.META_ARCHITECTURE == 'ContextUNet' and cfg.FINE_MODEL.CONTEXT_BLOCK is not None:
        #     cfg.DATA_LOADER.BATCH_SIZE = 3
        # elif cfg.FINE_MODEL.META_ARCHITECTURE == 'ContextUNet':
        #     cfg.DATA_LOADER.BATCH_SIZE = 4
        # else:
        #     cfg.DATA_LOADER.BATCH_SIZE = 5

        for fold in cfg.DATA_LOADER.FIVE_FOLD_LIST:
            cfg.DATA_LOADER.TRAIN_VAL_FOLD = fold
            cfg.DATA_LOADER.TRAIN_DB_FILE = train_db_path + str(fold)
            cfg.DATA_LOADER.VAL_DB_FILE = val_db_path + str(fold)

            cfg.ENVIRONMENT.PHASE = 'train'
            segmentation = SegmentationMultiProcess()
            run_multiprocessing(segmentation.run, cfg, cfg.ENVIRONMENT.NUM_GPU)

            cfg.TESTING.FINE_MODEL_WEIGHT_DIR = os.path.join(cfg.TRAINING.SAVER.SAVER_DIR,
                                                             str(cfg.ENVIRONMENT.EXPERIMENT_NAME) + '_' +
                                                             str(cfg.FINE_MODEL.META_ARCHITECTURE) + '_' +
                                                             str(cfg.FINE_MODEL.ENCODER_CONV_BLOCK) + '_' +
                                                             str(cfg.FINE_MODEL.DECODER_CONV_BLOCK) + '_' +
                                                             str(cfg.FINE_MODEL.CONTEXT_BLOCK) +
                                                             '_fine_size-' + str(cfg.FINE_MODEL.INPUT_SIZE[0]) +
                                                             '_channel-' + str(cfg.FINE_MODEL.NUM_CHANNELS[0]) +
                                                             '_depth-' + str(cfg.FINE_MODEL.NUM_DEPTH) +
                                                             '_loss-' + cfg.TRAINING.LOSS +
                                                             '_metric-' + cfg.TRAINING.METRIC) + '/fold_' \
                                                             + str(fold) + '/models/best_model.pt'
            cfg.ENVIRONMENT.PHASE = 'test'
            cfg.DATA_LOADER.TEST_DB_FILE = cfg.DATA_LOADER.VAL_DB_FILE
            cfg.TESTING.SAVER_DIR = './output/val'
            cfg.TESTING.IS_SAVE_MASK = False
            segmentation = SegmentationMultiProcess()
            run_multiprocessing(segmentation.run, cfg, 1)

            cfg.DATA_LOADER.TEST_DB_FILE = test_db_path
            cfg.TESTING.SAVER_DIR = './output/test'
            cfg.TESTING.IS_SAVE_MASK = True
            segmentation = SegmentationMultiProcess()
            run_multiprocessing(segmentation.run, cfg, 1)

    torch.cuda.synchronize()
    end_time = time.time()
    print('Training finish,  time: {}'.format(end_time))
    print('Training time: {} hours'.format((end_time - start_time) / 60 / 60))
