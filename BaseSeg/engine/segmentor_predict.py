
import os
import sys
import time
import datetime
import numpy as np
from multiprocessing import Process

import torch
import torch.nn.functional as F

from ..network.get_model import get_coarse_model, get_fine_model
from ..data.dataset_predict import test_collate_fn, PredictDataset, DataLoaderX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Common.logger import get_logger
from Common.image_io import save_ct_from_npy
from Common.image_resample import ScipyResample
from Common.image_process import change_axes_of_image, clip_and_normalize_mean_std
from Common.mask_process import extract_bbox, crop_image_according_to_bbox, extract_topk_largest_candidates


class SegmentationInfer(object):
    def __init__(self, cfg):
        super(SegmentationInfer, self).__init__()

        # step 1 >>> init params
        self.cfg = cfg
        self.save_dir = self.cfg.TESTING.SAVER_DIR
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # step 2 >>> init model
        self.coarse_model = get_coarse_model(cfg, 'test')
        self.fine_model = get_fine_model(cfg, 'test')
        if self.cfg.ENVIRONMENT.CUDA:
            # set fp16 inference
            if self.cfg.TESTING.IS_FP16:
                self.coarse_model = self.coarse_model.half()
                self.fine_model = self.fine_model.half()

            self.coarse_model = self.coarse_model.cuda()
            self.fine_model = self.fine_model.cuda()

        # step 3 >>> init log
        self.log_dir = './output/logs_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = get_logger(self.log_dir)
        self.logger.info('\n------------ {} options -------------'.format('test'))
        self.logger.info('%s' % str(self.cfg))
        self.logger.info('-------------- End ----------------\n')

        # step 4 >>> load model weight
        self._load_weights()
        self._set_requires_grad(self.coarse_model, False)
        self._set_requires_grad(self.fine_model, False)

    def _load_weights(self):
        self.coarse_model_weight_dir = self.cfg.TESTING.COARSE_MODEL_WEIGHT_DIR
        if self.coarse_model_weight_dir is not None and os.path.exists(self.coarse_model_weight_dir):
            checkpoint = torch.load(self.coarse_model_weight_dir)
            self.coarse_model.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            self.logger.info('load coarse model success!')
        else:
            raise ValueError('Does not exist coarse model weight path!')

        self.fine_model_weight_dir = self.cfg.TESTING.FINE_MODEL_WEIGHT_DIR
        if self.fine_model_weight_dir is not None and os.path.exists(self.fine_model_weight_dir):
            checkpoint = torch.load(self.fine_model_weight_dir)
            # self.fine_model.load_state_dict(
            #     {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            model_dict = self.fine_model.state_dict()
            pretrained_dict = checkpoint['state_dict']

            # filter out unnecessary keys
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
                               if k.replace('module.', '') in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.fine_model.load_state_dict(model_dict)
        else:
            raise ValueError('Does not exist fine model weight path!')

    @staticmethod
    def _set_requires_grad(model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def run(self):
        self.coarse_model.eval()
        self.fine_model.eval()

        if self.cfg.TESTING.IS_SYNCHRONIZATION:
            test_dataloader = PredictDataset(self.cfg)
        else:
            test_dataset = PredictDataset(self.cfg)
            test_dataloader = DataLoaderX(
                dataset=test_dataset,
                batch_size=self.cfg.TESTING.BATCH_SIZE,
                num_workers=self.cfg.TESTING.NUM_WORKER,
                shuffle=False,
                drop_last=False,
                collate_fn=test_collate_fn,
                pin_memory=True)

        self.logger.info('Starting test...')
        self.logger.info('test samples: {}'.format(len(test_dataloader)))

        torch.cuda.synchronize()
        t_start = time.time()
        for batch_idx, data_dict in enumerate(test_dataloader):
            self.logger.info('process: {}/{}'.format(batch_idx, len(test_dataloader)))

            data_dict = data_dict[0] if type(data_dict) is list else data_dict
            series_id = data_dict['series_id']
            raw_image = data_dict['image']
            raw_spacing = data_dict['raw_spacing']
            image_direction = data_dict['direction']
            coarse_image = data_dict['coarse_input_image']
            coarse_zoom_factor = data_dict['coarse_zoom_factor']

            # ----------------------------------------------------------------------------------------------------------
            # segmentation in coarse resolution.
            self.logger.info('coarse segmentation start...')
            coarse_image = torch.from_numpy(coarse_image[np.newaxis, np.newaxis]).float()
            if self.cfg.ENVIRONMENT.CUDA:
                coarse_image = coarse_image.half() if self.cfg.TESTING.IS_FP16 else coarse_image
                coarse_image = coarse_image.cuda()

            with torch.no_grad():
                coarse_image = self.coarse_model(coarse_image)

            coarse_image = coarse_image.cpu().float()
            torch.cuda.empty_cache()
            pred_coarse_mask = coarse_image
            pred_coarse_mask = F.sigmoid(pred_coarse_mask)
            pred_coarse_mask = torch.where(pred_coarse_mask >= 0.5, torch.tensor(1), torch.tensor(0))
            pred_coarse_mask = pred_coarse_mask.numpy().squeeze(axis=0).astype(np.uint8)

            # keep kth largest connected region.
            if self.cfg.TESTING.IS_POST_PROCESS:
                coarse_spacing = [raw_spacing[i]*coarse_zoom_factor[i] for i in range(3)]
                area_least = 1000 / coarse_spacing[0] / coarse_spacing[1] / coarse_spacing[2]
                out_coarse_mask = extract_topk_largest_candidates(pred_coarse_mask,
                                                                  self.cfg.DATA_LOADER.LABEL_NUM, area_least)
            else:
                coarse_image_shape = pred_coarse_mask.shape
                out_coarse_mask = np.zeros(coarse_image_shape[1:])
                for i in range(coarse_image_shape[0]):
                    out_coarse_mask[pred_coarse_mask[i] != 0] = i+1

            # crop image based coarse segmentation mask.
            coarse_bbox = extract_bbox(out_coarse_mask)
            raw_bbox = [int(coarse_bbox[0][0] * coarse_zoom_factor[0]),
                        int(coarse_bbox[0][1] * coarse_zoom_factor[0]),
                        int(coarse_bbox[1][0] * coarse_zoom_factor[1]),
                        int(coarse_bbox[1][1] * coarse_zoom_factor[1]),
                        int(coarse_bbox[2][0] * coarse_zoom_factor[2]),
                        int(coarse_bbox[2][1] * coarse_zoom_factor[2])]
            margin = [self.cfg.DATA_LOADER.EXTEND_SIZE / raw_spacing[i] for i in range(3)]
            crop_image, crop_fine_bbox = crop_image_according_to_bbox(raw_image, raw_bbox, margin)
            self.logger.info('coarse segmentation complete!')

            # ----------------------------------------------------------------------------------------------------------
            # segmentation in fine resolution.
            self.logger.info('fine segmentation start...')
            raw_image_shape = raw_image.shape
            crop_image_size = crop_image.shape
            fine_zoom_factor = crop_image_size / np.array(self.cfg.FINE_MODEL.INPUT_SIZE)
            if not self.cfg.FINE_MODEL.IS_PREPROCESS:
                crop_image, _ = ScipyResample.resample_to_size(crop_image, self.cfg.FINE_MODEL.INPUT_SIZE, order=1)
                crop_image = clip_and_normalize_mean_std(crop_image, self.cfg.DATA_LOADER.WINDOW_LEVEL[0],
                                                         self.cfg.DATA_LOADER.WINDOW_LEVEL[1])

            crop_image = crop_image.copy()
            crop_image = torch.from_numpy(crop_image[np.newaxis, np.newaxis]).float()
            if self.cfg.ENVIRONMENT.CUDA:
                crop_image = crop_image.half() if self.cfg.TESTING.IS_FP16 else crop_image
                crop_image = crop_image.cuda()

            with torch.no_grad():
                crop_image = self.fine_model(crop_image)

            crop_image = crop_image.cpu().float()
            torch.cuda.empty_cache()

            predict_fine_mask = crop_image
            predict_fine_mask = F.sigmoid(predict_fine_mask)

            if self.cfg.FINE_MODEL.IS_POSTPROCESS:
                predict_fine_mask = torch.where(predict_fine_mask >= 0.5, torch.tensor(1), torch.tensor(0))
            predict_fine_mask = predict_fine_mask.numpy().squeeze(axis=0)

            if self.cfg.TESTING.IS_SYNCHRONIZATION:
                self._post_process(batch_idx, series_id, predict_fine_mask, raw_image_shape, raw_spacing,
                                   crop_image_size, crop_fine_bbox, image_direction, fine_zoom_factor)
            else:
                p1 = Process(target=self._post_process,
                             args=(batch_idx, series_id, predict_fine_mask, raw_image_shape, raw_spacing,
                                   crop_image_size, crop_fine_bbox, image_direction, fine_zoom_factor))
                p1.daemon = False
                p1.start()

        torch.cuda.synchronize()
        t_end = time.time()
        average_time_usage = (t_end - t_start) * 1.0 / len(test_dataloader)
        time_score = (100-average_time_usage) * 1.0 / 100
        self.logger.info("Average time usage: {} s".format(average_time_usage))
        self.logger.info("Normalized time coefficient: {}".format(time_score))

    def _post_process(self, batch_idx, series_id, predict, raw_image_shape, raw_spacing,
                      crop_image_size, crop_fine_bbox, image_direction, fine_zoom_factor):
        self.logger.info('batch_id: {}, post process start...'.format(batch_idx))

        if not self.cfg.FINE_MODEL.IS_POSTPROCESS:
            fine_mask = []
            for i in range(len(predict)):
                mask, _ = ScipyResample.resample_to_size(predict[i], crop_image_size,
                                                         order=self.cfg.TESTING.OUT_RESAMPLE_MODE)
                fine_mask.append(mask)
            predict = fine_mask
            predict = np.stack(predict, axis=0)
            predict = np.where(predict >= 0.5, 1, 0)

        num_class = len(self.cfg.DATA_LOADER.LABEL_INDEX)
        if self.cfg.TESTING.IS_POST_PROCESS:
            fine_spacing = [raw_spacing[i]*fine_zoom_factor[i] for i in range(3)]
            area_least = 2000 / fine_spacing[0] / fine_spacing[1] / fine_spacing[2]
            predict = extract_topk_largest_candidates(predict, self.cfg.DATA_LOADER.LABEL_NUM, area_least)
        else:
            t_mask = np.zeros(predict.shape[1:], np.uint8)
            for i in range(num_class):
                t_mask[predict[i] != 0] = i + 1
            predict = t_mask

        out_mask = np.zeros(raw_image_shape, np.uint8)
        out_mask[crop_fine_bbox[0]:crop_fine_bbox[1],
                 crop_fine_bbox[2]:crop_fine_bbox[3],
                 crop_fine_bbox[4]:crop_fine_bbox[5]] = predict

        if self.cfg.TESTING.IS_SAVE_MASK:
            mask_path = os.path.join(self.save_dir, series_id + ".nii.gz")
            if raw_spacing[0] != raw_spacing[1]:
                raw_spacing = [raw_spacing[2], raw_spacing[1], raw_spacing[0]]
            out_mask = change_axes_of_image(out_mask, image_direction)
            save_ct_from_npy(out_mask, mask_path, spacing=raw_spacing, use_compression=True)
            self.logger.info('save fine mask complete!')

        self.logger.info('batch_id: {}, fine segmentation complete!'.format(batch_idx))