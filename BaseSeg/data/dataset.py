
import os
import sys
import json

import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from .data_prepare import MaskData

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.file_utils import read_txt
from Common.image_io import load_ct_info
from Common.image_augment import DataAugmentor
from Common.image_resample import ScipyResample
from Common.image_process import change_axes_of_image
from Common.mask_process import convert_mask_2_one_hot
from Common.mask_process import crop_image_according_to_mask
from Common.image_process import clip_and_normalize_mean_std, normalize_min_max_and_clip

from .data_prepare import generate_heatmap


def test_collate_fn(batch):
    return batch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SegDataSet(Dataset):

    def __init__(self, cfg, phase):
        super(SegDataSet, self).__init__()

        self.cfg = cfg
        self.phase = phase

        if self.phase == 'val':
            self.db_file = self.cfg.DATA_LOADER.VAL_DB_FILE
        elif self.phase == 'test':
            self.db_file = self.cfg.DATA_LOADER.TEST_DB_FILE
        else:
            self.db_file = self.cfg.DATA_LOADER.TRAIN_DB_FILE

        self.label = cfg.DATA_LOADER.LABEL_INDEX
        self.is_coarse = cfg.DATA_LOADER.IS_COARSE
        self.window_level = cfg.DATA_LOADER.WINDOW_LEVEL
        self.is_data_augment = cfg.DATA_AUGMENT.IS_ENABLE

        self.data_info = self._read_db()

    def __len__(self):
        if self.cfg.ENVIRONMENT.IS_SMOKE_TEST:
            return len(self.data_info[:4])
        else:
            return len(self.data_info)

    def __getitem__(self, idx):
        """
        Return:
            image(torch tensor): channel first, dims=[c, z, y, x]
            mask(torch tensor): channel first, dims=[c, z, y, x]
        """
        data = self.data_info[idx]
        if self.phase == 'test':
            image_path = data.image_path
            mask_path = data.mask_path
            image_dict = load_ct_info(image_path)
            raw_image = image_dict['npy_image']
            raw_spacing = image_dict['spacing']
            image_direction = image_dict['direction']
            if self.cfg.DATA_LOADER.IS_NORMALIZATION_DIRECTION:
                raw_image = change_axes_of_image(raw_image, image_direction)
            if mask_path is not None and os.path.exists(mask_path):
                mask_dict = load_ct_info(mask_path)
                raw_mask = mask_dict['npy_image']
                if self.cfg.DATA_LOADER.IS_NORMALIZATION_DIRECTION:
                    raw_mask = change_axes_of_image(raw_mask, image_direction)
                raw_mask = convert_mask_2_one_hot(raw_mask, self.label)
            else:
                raw_mask = None

            if not self.cfg.COARSE_MODEL.IS_PREPROCESS:
                zoom_image, zoom_factor = ScipyResample.resample_to_size(raw_image, self.cfg.COARSE_MODEL.INPUT_SIZE)
                if self.cfg.DATA_LOADER.IS_NORMALIZATION_HU:
                    norm_image = clip_and_normalize_mean_std(zoom_image, self.window_level[0], self.window_level[1])
                else:
                    norm_image = normalize_min_max_and_clip(zoom_image, self.window_level[0], self.window_level[1])
            else:
                norm_image = raw_image.copy()
                source_size = raw_image.shape
                zoom_factor = source_size / np.array(self.cfg.COARSE_MODEL.INPUT_SIZE)

            return {'series_id': data.series_id,
                    'image': raw_image,
                    'raw_spacing': raw_spacing,
                    'raw_mask': raw_mask,
                    'image_direction': image_direction,
                    'coarse_input_image': norm_image,
                    'coarse_zoom_factor': zoom_factor}
        elif self.phase == 'val' or not self.is_data_augment:
            if self.is_coarse:
                image_path = data.coarse_image_path
                mask_path = data.coarse_mask_path
            else:
                image_path = data.fine_image_path
                mask_path = data.fine_mask_path

            npy_image = np.load(image_path + '.npy')
            npy_mask = np.load(mask_path + '.npy')
        else:
            image_path = data.image_path
            mask_path = data.mask_path

            image_dict = load_ct_info(image_path)
            npy_image = image_dict['npy_image']
            image_spacing = image_dict['spacing']
            mask_dict = load_ct_info(mask_path)
            npy_mask = mask_dict['npy_image']

            image_direction = image_dict['direction']
            if self.cfg.DATA_LOADER.IS_NORMALIZATION_DIRECTION:
                npy_image = change_axes_of_image(npy_image, image_direction)
                npy_mask = change_axes_of_image(npy_mask, image_direction)

            data_augmentor = DataAugmentor()
            if self.cfg.DATA_AUGMENT.IS_RANDOM_ROTATE:
                npy_image, npy_mask = data_augmentor.random_rotate(npy_image, npy_mask,
                                                                   min_angle=self.cfg.DATA_AUGMENT.ROTATE_ANGLE[0],
                                                                   max_angle=self.cfg.DATA_AUGMENT.ROTATE_ANGLE[1])

            if self.cfg.DATA_LOADER.IS_COARSE:
                if self.cfg.DATA_AUGMENT.IS_RANDOM_SHIFT and np.random.choice([0, 1]):
                    npy_image, npy_mask = data_augmentor.random_shift(npy_image, npy_mask,
                                                                      self.cfg.DATA_AUGMENT.SHIFT_MAX_RATIO)
                out_size = self.cfg.COARSE_MODEL.INPUT_SIZE
            else:
                if self.cfg.DATA_AUGMENT.IS_RANDOM_CROP_TO_LABELS and np.random.choice([0, 1]):
                    margin = [int(self.cfg.DATA_AUGMENT.MAX_EXTEND_SIZE / image_spacing[0]),
                              int(self.cfg.DATA_AUGMENT.MAX_EXTEND_SIZE / image_spacing[1]),
                              int(self.cfg.DATA_AUGMENT.MAX_EXTEND_SIZE / image_spacing[2])]
                    npy_image, npy_mask = data_augmentor.random_crop_to_extend_labels(npy_image, npy_mask, margin)
                else:
                    margin = [int(self.cfg.DATA_LOADER.EXTEND_SIZE / image_spacing[0]),
                              int(self.cfg.DATA_LOADER.EXTEND_SIZE / image_spacing[1]),
                              int(self.cfg.DATA_LOADER.EXTEND_SIZE / image_spacing[2])]
                    npy_image, npy_mask = crop_image_according_to_mask(npy_image, npy_mask, margin)
                out_size = self.cfg.FINE_MODEL.INPUT_SIZE
            npy_image, _ = ScipyResample.resample_to_size(npy_image, out_size)
            npy_mask, _ = ScipyResample.resample_mask_to_size(npy_mask, out_size,
                                                              num_label=np.max(np.array(self.label)))

            if self.cfg.DATA_AUGMENT.IS_ELASTIC_TRANSFORM and np.random.choice([0, 1]):
                npy_image, npy_mask = data_augmentor.elastic_transform_3d(npy_image, npy_mask)

            if self.cfg.DATA_AUGMENT.IS_CHANGE_ROI_HU:
                npy_image = data_augmentor.augment_brightness_additive(npy_image, npy_mask,
                                                                       self.cfg.DATA_LOADER.LABEL_INDEX,
                                                                       self.cfg.DATA_AUGMENT.ROI_HU_RANGE)

            if self.cfg.DATA_AUGMENT.IS_RANDOM_FLIP:
                npy_image, npy_mask = data_augmentor.random_flip(npy_image, npy_mask)

        if self.cfg.DATA_LOADER.IS_NORMALIZATION_HU:
            npy_image = clip_and_normalize_mean_std(npy_image, self.window_level[0], self.window_level[1])
        else:
            npy_image = normalize_min_max_and_clip(npy_image, self.window_level[0], self.window_level[1])

        if self.phase == 'train' and self.is_data_augment and \
                self.cfg.DATA_AUGMENT.IS_ADD_GAUSSIAN_NOISE and np.random.choice([0, 1]):
            data_augmentor = DataAugmentor()
            npy_image = data_augmentor.augment_gaussian_noise(npy_image, noise_variance=(0, 0.1))

        image_czyx = npy_image[np.newaxis, ]
        if self.cfg.DATA_LOADER.IS_COARSE and self.cfg.COARSE_MODEL.NUM_CLASSES == 1:
            npy_mask[npy_mask != 0] = 1
            mask_czyx = npy_mask[np.newaxis, ].astype(np.uint8)
        else:
            mask_czyx = convert_mask_2_one_hot(npy_mask, self.label)

        if self.cfg.FINE_MODEL.AUXILIARY_TASK:
            is_single_channel = True if self.cfg.FINE_MODEL.AUXILIARY_CLASS == 1 else False
            heat_maps = generate_heatmap(mask_czyx, self.cfg.DATA_LOADER.LABEL_NUM, sigma=7, stride=4,
                                         is_single_channel=is_single_channel)
            return torch.from_numpy(image_czyx).float(), \
                   torch.from_numpy(mask_czyx).float(), \
                   torch.from_numpy(heat_maps).float()
        else:
            return torch.from_numpy(image_czyx).float(), torch.from_numpy(mask_czyx).float()

    def _read_db(self):
        bad_case_path = self.cfg.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT
        if self.phase == 'train' and bad_case_path is not None and os.path.exists(bad_case_path):
            bad_case_uids = read_txt(bad_case_path)
        else:
            bad_case_uids = None
        local_data = []
        env = lmdb.open(self.db_file, map_size=int(1e9))
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            tmp_data = MaskData(series_id=key,
                                image_path=label_info['image_path'],
                                mask_path=label_info['mask_path'],
                                coarse_image_path=label_info['coarse_image_path']
                                if 'coarse_image_path' in label_info else None,
                                coarse_mask_path=label_info['coarse_mask_path']
                                if 'coarse_mask_path' in label_info else None,
                                fine_image_path=label_info['fine_image_path']
                                if 'fine_image_path' in label_info else None,
                                fine_mask_path=label_info['fine_mask_path']
                                if 'fine_mask_path' in label_info else None)
            if bad_case_uids is not None and tmp_data.series_id in bad_case_uids:
                local_data.extend([tmp_data]*self.cfg.DATA_LOADER.BAD_CASE_AUGMENT_TIMES)

            local_data.append(tmp_data)
        env.close()

        return local_data