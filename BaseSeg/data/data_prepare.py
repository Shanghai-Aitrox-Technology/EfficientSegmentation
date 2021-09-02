
import os
import sys
import json
import traceback

import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from Common.image_io import load_ct_info, save_ct_from_npy
from Common.lmdb_io import DataBaseUtils
from Common.image_resample import ScipyResample
from Common.file_utils import MyEncoder, read_txt
from Common.image_process import change_axes_of_image
from Common.mask_process import smooth_mask, crop_image_according_to_mask, extract_candidate_centroid


def run_prepare_data(cfg):
    data_prepare = DataPrepare(cfg)

    pool = Pool(int(cpu_count() * 0.7))
    for data in data_prepare.data_info:
        try:
            pool.apply_async(data_prepare.process, (data,))
        except Exception as err:
            traceback.print_exc()
            print('Create coarse/fine image/mask throws exception %s, with series_id %s!' % (err, data.series_id))

    pool.close()
    pool.join()

    data_prepare._split_train_val()


class MaskData(object):
    def __init__(self, series_id, image_path, mask_path,
                 smooth_mask_path=None, coarse_image_path=None,
                 coarse_mask_path=None, fine_image_path=None, fine_mask_path=None):
        super(MaskData, self).__init__()

        self.series_id = series_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.smooth_mask_path = smooth_mask_path
        self.coarse_image_path = coarse_image_path
        self.coarse_mask_path = coarse_mask_path
        self.fine_image_path = fine_image_path
        self.fine_mask_path = fine_mask_path


class DataPrepare(object):
    def __init__(self, cfg):
        super(DataPrepare, self).__init__()
        self.cfg = cfg
        self.out_dir = cfg.DATA_PREPARE.OUT_DIR
        self.db_dir = self.out_dir + '/db/'
        self.train_db_file = self.db_dir + 'seg_raw_train'
        self.test_db_file = self.db_dir + 'seg_raw_test'
        self.out_db_file = self.db_dir + 'seg_pre-process_database'
        self.out_train_db_file = self.db_dir + 'seg_train_fold_1'
        self.out_val_db_file = self.db_dir + 'seg_val_fold_1'

        self.image_dir = cfg.DATA_PREPARE.TRAIN_IMAGE_DIR
        self.mask_dir = cfg.DATA_PREPARE.TRAIN_MASK_DIR
        self.mask_label = cfg.DATA_PREPARE.MASK_LABEL
        self.extend_size = cfg.DATA_PREPARE.EXTEND_SIZE

        self.out_coarse_size = cfg.DATA_PREPARE.OUT_COARSE_SIZE
        self.out_coarse_spacing = cfg.DATA_PREPARE.OUT_COARSE_SPACING
        self.out_fine_size = cfg.DATA_PREPARE.OUT_FINE_SIZE
        self.out_fine_spacing = cfg.DATA_PREPARE.OUT_FINE_SPACING

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self._create_db_file(phase='train')
        self._create_db_file(phase='test')
        self.data_info = self._read_db()

        # # create dir to save smooth mask
        # self.smooth_mask_save_dir = os.path.join(self.out_dir, 'smooth_mask')
        # if not os.path.exists(self.smooth_mask_save_dir):
        #     os.makedirs(self.smooth_mask_save_dir)

        # create dir to save coarse image and mask
        if (self.out_coarse_size is not None and self.out_coarse_spacing is not None) or (
                self.out_coarse_size is None and self.out_coarse_spacing is None):
            print('One and just one can be set not none between out_coarse_size and out_coarse_spacing!')
            return

        if self.out_coarse_size is not None:
            coarse_prefix = '{}_{}_{}'.format(self.out_coarse_size[0],
                                              self.out_coarse_size[1],
                                              self.out_coarse_size[2])
        else:
            coarse_prefix = '{}_{}_{}'.format(self.out_coarse_spacing[0],
                                              self.out_coarse_spacing[1],
                                              self.out_coarse_spacing[2])

        self.coarse_image_save_dir = os.path.join(self.out_dir, 'coarse_image', coarse_prefix)
        if not os.path.exists(self.coarse_image_save_dir):
            os.makedirs(self.coarse_image_save_dir)

        self.coarse_mask_save_dir = os.path.join(self.out_dir, 'coarse_mask', coarse_prefix)
        if not os.path.exists(self.coarse_mask_save_dir):
            os.makedirs(self.coarse_mask_save_dir)

        # create dir to save fine image and mask
        if (self.out_fine_size is not None and self.out_fine_spacing is not None) or (
                self.out_fine_size is None and self.out_fine_spacing is None):
            print('One and just one can be set not none between out_fine_size and out_fine_spacing!')
            return

        if self.out_fine_size is not None:
            fine_prefix = '{}_{}_{}'.format(self.out_fine_size[0],
                                            self.out_fine_size[1],
                                            self.out_fine_size[2])
        else:
            fine_prefix = '{}_{}_{}'.format(self.out_fine_spacing[0],
                                            self.out_fine_spacing[1],
                                            self.out_fine_spacing[2])

        self.fine_image_save_dir = os.path.join(self.out_dir, 'fine_image', fine_prefix)
        if not os.path.exists(self.fine_image_save_dir):
            os.makedirs(self.fine_image_save_dir)

        self.fine_mask_save_dir = os.path.join(self.out_dir, 'fine_mask', fine_prefix)
        if not os.path.exists(self.fine_mask_save_dir):
            os.makedirs(self.fine_mask_save_dir)

    def process(self, data):
        series_id = data.series_id
        image_info = load_ct_info(data.image_path)
        mask_info = load_ct_info(data.mask_path)

        print('Start processing %s.' % series_id)

        npy_image = image_info['npy_image']
        image_direction = image_info['direction']
        image_spacing = image_info['spacing']

        npy_mask = mask_info['npy_image']
        mask_direction = mask_info['direction']
        mask_spacing = mask_info['spacing']

        if self.cfg.DATA_PREPARE.IS_NORMALIZATION_DIRECTION:
            npy_image = change_axes_of_image(npy_image, image_direction)
            npy_mask = change_axes_of_image(npy_mask, mask_direction)

        if npy_image.shape != npy_mask.shape:
            print('Shape of image/mask are not equal in series_id: {}'.format(data.series_id))
            return

        num_label = np.max(np.array(self.mask_label))

        # # Process and save smooth mask
        # if self.cfg.DATA_PREPARE.IS_SMOOTH_MASK:
        #     data.smooth_mask_path = os.path.join(self.smooth_mask_save_dir, series_id)
        #     if os.path.exists(data.smooth_mask_path):
        #         npy_mask = np.load(data.smooth_mask_path)
        #     else:
        #         t_smooth_mask = np.zeros_like(npy_mask)
        #
        #         for i in range(1, num_label + 1):
        #             if i not in np.array(self.mask_label):
        #                 continue
        #
        #             t_mask = npy_mask.copy()
        #             t_mask = np.where(t_mask == i, 1, 0)
        #             area_least = int(2000 / mask_spacing[0] / mask_spacing[1] / mask_spacing[2])
        #             smooth_mask(t_mask, t_smooth_mask, out_num_label=i, area_least=area_least, is_binary_close=False)
        #         npy_mask = t_smooth_mask.copy()
        #         np.save(data.smooth_mask_path, t_smooth_mask)

        # Process and save coarse image and mask
        if self.out_coarse_size is not None:
            coarse_image, _ = ScipyResample.resample_to_size(npy_image, self.out_coarse_size)
            coarse_mask, _ = ScipyResample.resample_mask_to_size(npy_mask, self.out_coarse_size, num_label=num_label)
        else:
            coarse_image, _ = ScipyResample.resample_to_spacing(npy_image, image_spacing, self.out_coarse_spacing)
            coarse_mask, _ = ScipyResample.resample_mask_to_spacing(npy_mask, mask_spacing, self.out_coarse_spacing,
                                                                    num_label=num_label)

        data.coarse_image_path = os.path.join(self.coarse_image_save_dir, series_id)
        data.coarse_mask_path = os.path.join(self.coarse_mask_save_dir, series_id)
        np.save(data.coarse_image_path, coarse_image)
        np.save(data.coarse_mask_path, coarse_mask)

        # Process and save fine image and mask
        margin = [int(self.extend_size / image_spacing[0]),
                  int(self.extend_size / image_spacing[1]),
                  int(self.extend_size / image_spacing[2])]
        t_crop_image, t_crop_mask = crop_image_according_to_mask(npy_image, npy_mask, margin)

        if self.out_fine_size is not None:
            fine_image, _ = ScipyResample.resample_to_size(t_crop_image, self.out_fine_size)
            fine_mask, _ = ScipyResample.resample_mask_to_size(t_crop_mask, self.out_fine_size, num_label=num_label)
        else:
            fine_image, _ = ScipyResample.resample_to_spacing(t_crop_image, image_spacing, self.out_fine_spacing)
            fine_mask, _ = ScipyResample.resample_mask_to_spacing(t_crop_mask, mask_spacing, self.out_fine_spacing,
                                                                  num_label=num_label)

        data.fine_image_path = os.path.join(self.fine_image_save_dir, series_id)
        data.fine_mask_path = os.path.join(self.fine_mask_save_dir, series_id)
        np.save(data.fine_image_path, fine_image)
        np.save(data.fine_mask_path, fine_mask)

        # Update db
        self._update_db(data)

        print('End processing %s.' % series_id)

    def _read_db(self):
        local_data = []
        env = lmdb.open(self.train_db_file, map_size=int(1e9))
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            tmp_data = MaskData(key,
                                label_info['image_path'],
                                label_info['mask_path'])
            local_data.append(tmp_data)
        env.close()

        print('Num of ct is %d.' % len(local_data))
        return local_data

    def _update_db(self, data):
        env = lmdb.open(self.out_db_file, map_size=int(1e9))
        txn = env.begin(write=True)

        data_dict = {'image_path': data.image_path,
                     'mask_path': data.mask_path,
                     'smooth_mask_path': data.smooth_mask_path,
                     'coarse_image_path': data.coarse_image_path,
                     'coarse_mask_path': data.coarse_mask_path,
                     'fine_image_path': data.fine_image_path,
                     'fine_mask_path': data.fine_mask_path}

        txn.put(str(data.series_id).encode(), value=json.dumps(data_dict, cls=MyEncoder).encode())

        txn.commit()
        env.close()

    def _create_db_file(self, phase='train'):
        db_file_path = self.train_db_file if phase == 'train' else self.test_db_file
        DataBaseUtils.creat_db(db_file_path)

        series_ids = read_txt(self.cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT) \
            if phase == 'train' else read_txt(self.cfg.DATA_PREPARE.TEST_SERIES_IDS_TXT)

        for series_id in series_ids:
            if phase == 'train':
                image_path = os.path.join(self.cfg.DATA_PREPARE.TRAIN_IMAGE_DIR, series_id + '_0000.nii.gz')
                mask_path = os.path.join(self.cfg.DATA_PREPARE.TRAIN_MASK_DIR, series_id + '.nii.gz')
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    data_dict = {'image_path': image_path,
                                 'mask_path': mask_path}
                    DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)
                else:
                    print('%s has invalid image/mask.' % series_id)
            else:
                image_path = os.path.join(self.cfg.DATA_PREPARE.TEST_IMAGE_DIR, series_id + '_0000.nii.gz')
                mask_path = os.path.join(self.cfg.DATA_PREPARE.TEST_MASK_DIR, series_id + '.nii.gz') \
                    if self.cfg.DATA_PREPARE.TEST_MASK_DIR is not None else None
                if os.path.exists(image_path):
                    data_dict = {'image_path': image_path,
                                 'mask_path': mask_path}
                    DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)
                else:
                    print('%s has invalid image.' % series_id)

    def _split_train_val(self):
        default_train_db = self.cfg.DATA_PREPARE.DEFAULT_TRAIN_DB
        default_val_db = self.cfg.DATA_PREPARE.DEFAULT_VAL_DB

        if default_train_db is not None and default_val_db is not None:
            env = lmdb.open(default_train_db, map_size=int(1e9))
            txn = env.begin()
            series_ids_train = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids_train.append(key)

            env = lmdb.open(default_val_db, map_size=int(1e9))
            txn = env.begin()
            series_ids_val = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids_val.append(key)

            env = lmdb.open(self.out_db_file, map_size=int(1e9))
            txn = env.begin()
        else:
            env = lmdb.open(self.out_db_file, map_size=int(1e9))
            txn = env.begin()
            series_ids = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids.append(key)

            series_ids_train, series_ids_val = train_test_split(series_ids, test_size=self.cfg.DATA_PREPARE.VAL_RATIO,
                                                                random_state=0)

        print('Num of train series is: %d, num of val series is: %d.' % (len(series_ids_train), len(series_ids_val)))

        # create train db
        env_train = lmdb.open(self.out_train_db_file, map_size=int(1e9))
        txn_train = env_train.begin(write=True)
        for series_id in series_ids_train:
            value = str(txn.get(str(series_id).encode()), encoding='utf-8')
            txn_train.put(key=str(series_id).encode(), value=str(value).encode())
        txn_train.commit()
        env_train.close()

        # create val db
        env_val = lmdb.open(self.out_val_db_file, map_size=int(1e9))
        txn_val = env_val.begin(write=True)
        for series_id in series_ids_val:
            value = str(txn.get(str(series_id).encode()), encoding='utf-8')
            txn_val.put(key=str(series_id).encode(), value=str(value).encode())
        txn_val.commit()
        env_val.close()

        env.close()
        if self.cfg.DATA_PREPARE.IS_SPLIT_5FOLD:
            self._split_5fold_train_val()

    def _split_5fold_train_val(self):
        raw_train_db = self.out_train_db_file
        raw_val_db = self.out_val_db_file
        new_train_db = raw_train_db.split('_1')[0]
        new_val_db = raw_val_db.split('_1')[0]

        env = lmdb.open(raw_train_db, map_size=int(1e9))
        txn = env.begin()
        default_train_series_uid = []
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            default_train_series_uid.append(key)
        num_train = len(default_train_series_uid)
        new_train_series_uid = [default_train_series_uid[:int(num_train * 0.25)],
                                default_train_series_uid[int(num_train * 0.25):int(num_train * 0.5)],
                                default_train_series_uid[int(num_train * 0.5):int(num_train * 0.75)],
                                default_train_series_uid[int(num_train * 0.75):]]

        env = lmdb.open(raw_val_db, map_size=int(1e9))
        txn = env.begin()
        default_val_series_uid = []
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            default_val_series_uid.append(key)

        env = lmdb.open(self.out_db_file, map_size=int(1e9))
        txn = env.begin()

        for i in range(4):
            out_train_db = new_train_db + '_' + str(i + 2)
            out_val_db = new_val_db + '_' + str(i + 2)
            out_5fold_train = []
            for j in range(4):
                if i != j:
                    out_5fold_train.extend(new_train_series_uid[j])
            out_5fold_train.extend(default_val_series_uid)
            out_5fold_val = new_train_series_uid[i]

            # create train db
            env_train = lmdb.open(out_train_db, map_size=int(1e9))
            txn_train = env_train.begin(write=True)
            for series_id in out_5fold_train:
                value = str(txn.get(str(series_id).encode()), encoding='utf-8')
                txn_train.put(key=str(series_id).encode(), value=str(value).encode())
            txn_train.commit()
            env_train.close()

            # create val db
            env_val = lmdb.open(out_val_db, map_size=int(1e9))
            txn_val = env_val.begin(write=True)
            for series_id in out_5fold_val:
                value = str(txn.get(str(series_id).encode()), encoding='utf-8')
                txn_val.put(key=str(series_id).encode(), value=str(value).encode())
            txn_val.commit()
            env_val.close()
        env.close()


def generate_heatmap(npy_mask, candidates_num, sigma=5, stride=4, is_single_channel=False):
    image_shape = npy_mask.shape
    channel = image_shape[0]
    image_size = image_shape[1:]

    image_size = np.array(image_size, dtype=np.int32)
    heatmap_size = image_size // stride

    candidates_centroid = []
    for i in range(channel):
        centroid = extract_candidate_centroid(npy_mask[i].copy(), 100, candidates_num[i])
        if len(centroid) != candidates_num[i]:
            centroid.extend([[0, 0, 0] for _ in range(candidates_num[i]-len(centroid))])
        candidates_centroid.append(centroid)

    num_joints = len(candidates_centroid)
    if is_single_channel:
        target = np.zeros((1, heatmap_size[0], heatmap_size[1], heatmap_size[2]), dtype=np.float32)
    else:
        target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1], heatmap_size[2]), dtype=np.float32)

    for joint_id in range(num_joints):
        joint_centroid = candidates_centroid[joint_id]
        for xyz_coord in joint_centroid:
            if xyz_coord[0] == 0 or xyz_coord[1] == 0 or xyz_coord[2] == 0:
                continue
            xyz_coord = [int(coord // stride) for coord in xyz_coord]
            mu_z, mu_y, mu_x = xyz_coord

            tmp_size = sigma // 2

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_z - tmp_size), int(mu_y - tmp_size), int(mu_x - tmp_size)]
            br = [int(mu_z + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_x + tmp_size + 1)]

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            z = y[:, :, np.newaxis]
            x0 = y0 = z0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * (tmp_size // 2) ** 2))
            # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_z = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            g_x = max(0, -ul[2]), min(br[2], heatmap_size[2]) - ul[2]

            # Image range
            img_z = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            img_x = max(0, ul[2]), min(br[2], heatmap_size[2])

            if is_single_channel:
                target[0, img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[
                                                                                     g_z[0]:g_z[1],
                                                                                     g_y[0]:g_y[1],
                                                                                     g_x[0]:g_x[1]]
            else:
                target[joint_id, img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[
                                                                                            g_z[0]:g_z[1],
                                                                                            g_y[0]:g_y[1],
                                                                                            g_x[0]:g_x[1]]

    return target