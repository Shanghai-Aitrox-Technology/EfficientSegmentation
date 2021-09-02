
import os
import sys
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Common.file_utils import read_txt
from Common.image_io import load_ct_info
from Common.image_resample import ScipyResample
from Common.image_process import change_axes_of_image, clip_and_normalize_mean_std


def test_collate_fn(batch):
    return batch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class PredictDataset(Dataset):
    def __init__(self, cfg):
        super(PredictDataset, self).__init__()
        self.cfg = cfg
        self.image_dir = cfg.DATA_LOADER.TEST_IMAGE_DIR
        self.data_filenames = os.listdir(self.image_dir)
        self.window_level = cfg.DATA_LOADER.WINDOW_LEVEL
        if self.cfg.DATA_LOADER.TEST_SERIES_IDS_TXT is not None and \
           os.path.exists(self.cfg.DATA_LOADER.TEST_SERIES_IDS_TXT):
            all_series_uid = read_txt(self.cfg.DATA_LOADER.TEST_SERIES_IDS_TXT)
            filenames = []
            for file_name in self.data_filenames:
                series_id = file_name.split('_0000.nii.gz')[0]
                if series_id in all_series_uid:
                    filenames.append(file_name)
            self.data_filenames = filenames

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        file_name = self.data_filenames[idx]
        image_path = self.image_dir + file_name
        series_id = file_name.split('_0000.nii.gz')[0]

        image_dict = load_ct_info(image_path)
        raw_image = image_dict['npy_image']
        raw_spacing = image_dict['spacing']
        image_direction = image_dict['direction']
        if self.cfg.DATA_LOADER.IS_NORMALIZATION_DIRECTION:
            raw_image = change_axes_of_image(raw_image, image_direction)

        if not self.cfg.COARSE_MODEL.IS_PREPROCESS:
            zoom_image, zoom_factor = ScipyResample.resample_to_size(raw_image, self.cfg.COARSE_MODEL.INPUT_SIZE)
            zoom_image = clip_and_normalize_mean_std(zoom_image, self.window_level[0], self.window_level[1])
        else:
            zoom_image = raw_image.copy()
            source_size = raw_image.shape
            zoom_factor = source_size / np.array(self.cfg.COARSE_MODEL.INPUT_SIZE)

        return {'series_id': series_id,
                'image': raw_image,
                'raw_spacing': raw_spacing,
                'direction': image_direction,
                'coarse_input_image': zoom_image,
                'coarse_zoom_factor': zoom_factor}