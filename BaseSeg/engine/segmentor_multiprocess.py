
import os
import sys
import time
import copy
import datetime
import numpy as np
from pynvml.smi import nvidia_smi

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from apex import amp
from apex.parallel import DistributedDataParallel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.logger import get_logger
from Common.image_io import save_ct_from_npy
from Common.image_resample import ScipyResample
from Common.file_utils import write_txt, write_csv
from Common.image_process import clip_and_normalize_mean_std, change_axes_of_image
from Common.mask_process import extract_bbox, crop_image_according_to_bbox, extract_topk_largest_candidates

from ..losses.get_loss import SegLoss
from ..network.get_model import get_coarse_model, get_fine_model
from ..evaluation.metric import DiceMetric, compute_flare_metric
from ..data.dataset import DataLoaderX, SegDataSet, test_collate_fn


class SegmentationMultiProcess(object):

    def __init__(self):
        """segmentation in multi-process mode"""
        super(SegmentationMultiProcess, self).__init__()

    def run(self, process_idx, cfg):
        self.cfg = cfg
        self.phase = cfg.ENVIRONMENT.PHASE

        if self.cfg.DATA_LOADER.IS_COARSE and self.phase == 'train':
            self.train_save_dir = os.path.join(self.cfg.TRAINING.SAVER.SAVER_DIR,
                                               str(self.cfg.ENVIRONMENT.EXPERIMENT_NAME) + '_' +
                                               str(self.cfg.COARSE_MODEL.META_ARCHITECTURE) + '_' +
                                               str(self.cfg.COARSE_MODEL.ENCODER_CONV_BLOCK) + '_' +
                                               str(self.cfg.COARSE_MODEL.DECODER_CONV_BLOCK) + '_' +
                                               str(self.cfg.COARSE_MODEL.CONTEXT_BLOCK) +
                                               '_coarse_size-' + str(self.cfg.COARSE_MODEL.INPUT_SIZE[0]) +
                                               '_channel-' + str(self.cfg.COARSE_MODEL.NUM_CHANNELS[0]) +
                                               '_depth-' + str(self.cfg.COARSE_MODEL.NUM_DEPTH) +
                                               '_loss-' + self.cfg.TRAINING.LOSS +
                                               '_metric-' + self.cfg.TRAINING.METRIC) + '/fold_' \
                                               + str(self.cfg.DATA_LOADER.TRAIN_VAL_FOLD)
        elif self.phase == 'train':
            self.train_save_dir = os.path.join(self.cfg.TRAINING.SAVER.SAVER_DIR,
                                               str(self.cfg.ENVIRONMENT.EXPERIMENT_NAME) + '_' +
                                               str(self.cfg.FINE_MODEL.META_ARCHITECTURE) + '_' +
                                               str(self.cfg.FINE_MODEL.ENCODER_CONV_BLOCK) + '_' +
                                               str(self.cfg.FINE_MODEL.DECODER_CONV_BLOCK) + '_' +
                                               str(self.cfg.FINE_MODEL.CONTEXT_BLOCK) +
                                               '_fine_size-' + str(self.cfg.FINE_MODEL.INPUT_SIZE[0]) +
                                               '_channel-' + str(self.cfg.FINE_MODEL.NUM_CHANNELS[0]) +
                                               '_depth-' + str(self.cfg.FINE_MODEL.NUM_DEPTH) +
                                               '_loss-' + self.cfg.TRAINING.LOSS +
                                               '_metric-' + self.cfg.TRAINING.METRIC) + '/fold_' \
                                               + str(self.cfg.DATA_LOADER.TRAIN_VAL_FOLD)
        if self.phase == 'test':
            self.test_save_dir = os.path.join(cfg.TESTING.SAVER_DIR,
                                              str(self.cfg.ENVIRONMENT.EXPERIMENT_NAME) + '_' +
                                              str(self.cfg.FINE_MODEL.META_ARCHITECTURE) + '_' +
                                              str(self.cfg.FINE_MODEL.ENCODER_CONV_BLOCK) + '_' +
                                              str(self.cfg.FINE_MODEL.DECODER_CONV_BLOCK) + '_' +
                                              str(self.cfg.FINE_MODEL.CONTEXT_BLOCK) +
                                              '_size-' + str(self.cfg.FINE_MODEL.INPUT_SIZE[0]) +
                                              '_channel-' + str(self.cfg.FINE_MODEL.NUM_CHANNELS[0]) +
                                              '_depth-' + str(self.cfg.FINE_MODEL.NUM_DEPTH) +
                                              '_loss-' + self.cfg.TRAINING.LOSS +
                                              '_metric-' + self.cfg.TRAINING.METRIC) + '/fold_' \
                                               + str(self.cfg.DATA_LOADER.TRAIN_VAL_FOLD)

        # step 1 >>> init params
        self.start_epoch = self.cfg.TRAINING.SCHEDULER.START_EPOCH
        if self.cfg.TRAINING.OPTIMIZER.LR > self.cfg.DATA_LOADER.BATCH_SIZE * 1e-3:
            self.lr = self.cfg.TRAINING.OPTIMIZER.LR * self.cfg.DATA_LOADER.BATCH_SIZE
        else:
            self.lr = self.cfg.TRAINING.OPTIMIZER.LR

        self.is_apex_train = self.cfg.TRAINING.IS_APEX_TRAIN
        self.is_distributed_train = self.cfg.TRAINING.IS_DISTRIBUTED_TRAIN
        if self.phase != 'train':
            self.is_apex_train = False
            self.is_distributed_train = False

        self.num_worker = self.cfg.DATA_LOADER.NUM_WORKER
        if self.cfg.DATA_LOADER.NUM_WORKER <= self.cfg.DATA_LOADER.BATCH_SIZE + 2:
            self.num_worker = self.cfg.DATA_LOADER.BATCH_SIZE + 2

        if self.cfg.ENVIRONMENT.IS_SMOKE_TEST:
            self.cfg.TRAINING.SCHEDULER.TOTAL_EPOCHS = 2

        # step 2 >>> init model
        if self.phase == 'test':
            self.coarse_model = get_coarse_model(cfg, self.phase)
            self.fine_model = get_fine_model(cfg, self.phase)

        else:
            if self.cfg.DATA_LOADER.IS_COARSE:
                self.model = get_coarse_model(cfg, self.phase)
            else:
                self.model = get_fine_model(cfg, self.phase)

        # set distribute training config
        if self.is_distributed_train:
            dist.init_process_group(backend='nccl',
                                    init_method=self.cfg.ENVIRONMENT.INIT_METHOD,
                                    world_size=self.cfg.ENVIRONMENT.NUM_GPU,
                                    rank=process_idx)
            # local_rank = torch.distributed.get_rank()
            local_rank = process_idx
            torch.cuda.set_device(local_rank)
            self.device = torch.device('cuda', local_rank)

            self.local_rank = local_rank
            self.is_print_out = True if local_rank == 0 else False  # Only GPU 0 print information.
            if cfg.ENVIRONMENT.CUDA:
                self.model = self.model.to(self.device)
        else:
            self.is_print_out = True
            if cfg.ENVIRONMENT.CUDA:
                if self.phase == 'test':
                    if self.cfg.TESTING.IS_FP16:
                        self.coarse_model = self.coarse_model.half()
                        self.fine_model = self.fine_model.half()
                    self.coarse_model = self.coarse_model.cuda()
                    self.fine_model = self.fine_model.cuda()
                else:
                    self.model = self.model.cuda()

        # init optimizer
        self.optimizer = self._init_optimizer() if self.phase == 'train' else None

        # set apex training
        if self.is_apex_train:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        # load model weight
        self._load_weights()

        # # set cuda
        # if cfg.ENVIRONMENT.CUDA:
        #     cudnn.benchmark = True
        #     cudnn.deterministic = True
        #     cudnn.enabled = True

        # step 3 >>> init logger
        self.log_dir = os.path.join(self.train_save_dir, 'logs') if self.phase == 'train' else \
            os.path.join(self.test_save_dir, 'logs')
        if self.is_print_out and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.is_print_out:
            self.logger = get_logger(self.log_dir)
            self.logger.info('\n------------ {} options -------------'.format(self.phase))
            self.logger.info('%s' % str(self.cfg))
            self.logger.info('-------------- End ----------------\n')

        if self.phase == 'train':
            self.do_train(self.cfg.DATA_LOADER.TRAIN_VAL_FOLD)
        elif self.phase == 'test':
            self.do_test(self.cfg.DATA_LOADER.TRAIN_VAL_FOLD)

    def _init_optimizer(self):
        if self.cfg.TRAINING.OPTIMIZER.METHOD == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.TRAINING.OPTIMIZER.LR,
                                  momentum=0.99, weight_decay=self.cfg.TRAINING.OPTIMIZER.L2_PENALTY)
        elif self.cfg.TRAINING.OPTIMIZER.METHOD == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.TRAINING.OPTIMIZER.LR,
                                   betas=(0.9, 0.99), weight_decay=self.cfg.TRAINING.OPTIMIZER.L2_PENALTY)
        return optimizer

    def _load_weights(self):
        if self.phase == 'test':
            self.coarse_model_weight_dir = self.cfg.TESTING.COARSE_MODEL_WEIGHT_DIR
            self.fine_model_weight_dir = self.cfg.TESTING.FINE_MODEL_WEIGHT_DIR
            if self.coarse_model_weight_dir is not None and os.path.exists(self.coarse_model_weight_dir):
                checkpoint = torch.load(self.coarse_model_weight_dir)
                self.coarse_model.load_state_dict(
                    {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            else:
                # raise Warning('Does not exist the coarse model weight path!')
                print('Does not exist the coarse model weight path!')
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
                # raise Warning('Does not exist the fine model weight path!')
                print('Does not exist the fine model weight path!')
        else:
            self.weight_dir = self.cfg.COARSE_MODEL.WEIGHT_DIR if self.cfg.DATA_LOADER.IS_COARSE else \
                self.cfg.FINE_MODEL.WEIGHT_DIR
            if self.weight_dir is not None and os.path.exists(self.weight_dir):
                if self.is_print_out:
                    print('Loading pre_trained model...')
                checkpoint = torch.load(self.weight_dir)
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
                self.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
                self.lr = checkpoint['lr']
            else:
                if self.is_print_out:
                    print('Failed to load pre-trained network.')

    def _save_weights(self, epoch, net_state_dict, optimizer_state_dict):
        if self.is_print_out:
            model_dir = os.path.join(self.train_save_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.best_model_weight_path = os.path.join(model_dir, 'best_model.pt')

            torch.save({
                'epoch': epoch,
                'state_dict': net_state_dict,
                'optimizer_dict': optimizer_state_dict,
                'lr': self.lr},
                self.best_model_weight_path)

    def _get_lr_scheduler(self):
        if self.cfg.TRAINING.SCHEDULER.LR_SCHEDULE == "cosineLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=3e-5)
        elif self.cfg.TRAINING.SCHEDULER.LR_SCHEDULE == "stepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.5)

    @staticmethod
    def _set_requires_grad(model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def _get_lr(epoch, num_epochs, init_lr):
        if epoch <= num_epochs * 0.66:
            lr = init_lr
        elif epoch <= num_epochs * 0.86:
            lr = init_lr * 0.1
        else:
            lr = init_lr * 0.05

        return lr

    @staticmethod
    def _get_rank():
        """get gpu id in distribution training."""
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def _reduce_tensor(tensor: torch.Tensor):
        torch.distributed.barrier()
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
        rt /= torch.distributed.get_world_size()
        return rt

    @staticmethod
    def _cleanup_multiprocessing():
        dist.destroy_process_group()

    def _create_train_data(self):
        train_dataset = SegDataSet(self.cfg, 'train')
        val_dataset = SegDataSet(self.cfg, 'val')

        if self.is_distributed_train:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=self.cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=self.num_worker,
            shuffle=True if train_sampler is None else False,
            drop_last=False,
            pin_memory=True,
            sampler=train_sampler)

        self.val_loader = DataLoaderX(
            dataset=val_dataset,
            batch_size=self.cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            sampler=val_sampler)

    @staticmethod
    def get_gpu_memory_usage(time_interval, is_ongoing_monitor=True, gpu_index=0, out_path=None):
        gpu_memory_max = 0
        memory_metric_min = 1.0
        interval_nums = 0
        while is_ongoing_monitor:
            interval_nums += 1
            nvsmi = nvidia_smi.getInstance()
            dictm = nvsmi.DeviceQuery('memory.free, memory.total')
            gpu_memory = dictm['gpu'][int(gpu_index)]['fb_memory_usage']['total'] - \
                         dictm['gpu'][int(gpu_index)]['fb_memory_usage']['free']
            gpu_memory_total = dictm['gpu'][int(gpu_index)]['fb_memory_usage']['total']
            gpu_memory_max = max(gpu_memory_max, gpu_memory)
            memory_metric_min = min((gpu_memory_total-gpu_memory_max)*1.0/gpu_memory_total, memory_metric_min)
            if is_ongoing_monitor:
                time.sleep(time_interval)
                if interval_nums % 20 == 0:
                    if out_path is not None:
                        write_txt(out_path, [str(gpu_memory_max), str(memory_metric_min)])

        return gpu_memory_max, memory_metric_min

    def do_train(self, fold=1):
        if self.is_print_out:
            self.logger.info("start training {}th fold...".format(fold))
        self._create_train_data()

        if self.is_print_out:
            self.logger.info('Preprocess parallels: {}'.format(self.num_worker))
            self.logger.info('train samples per epoch: {}'.format(len(self.train_loader)))
            self.logger.info('val samples per epoch: {}'.format(len(self.val_loader)))

            self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            self.val_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))

        if torch.cuda.device_count() > 1:
            if not self.is_distributed_train:
                self.model = torch.nn.DataParallel(self.model)
            elif self.is_apex_train and self.is_distributed_train:
                self.model = DistributedDataParallel(self.model, delay_allreduce=True)
            elif self.is_distributed_train:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                       device_ids=[self.local_rank],
                                                                       output_device=self.local_rank)

        best_dice = 0

        for epoch in range(self.start_epoch, self.cfg.TRAINING.SCHEDULER.TOTAL_EPOCHS + 1):
            if self.is_print_out:
                self.logger.info('\nStarting training epoch {}'.format(epoch))

            start_time = datetime.datetime.now()
            self._train(epoch)
            val_dice = self._validate(epoch)

            if self.is_print_out:
                if val_dice > best_dice:
                    best_dice = val_dice
                    self._save_weights(epoch, self.model.state_dict(), self.optimizer.state_dict())

                self.logger.info('End of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - start_time))

        if self.is_print_out:
            self.train_writer.close()
            self.val_writer.close()
            self.logger.info('\nEnd of training, best dice: {}'.format(best_dice))
            self.logger.info("Training {}th fold finish!".format(fold))

    def _train(self, epoch):
        self.model.train()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr(epoch, self.cfg.TRAINING.SCHEDULER.TOTAL_EPOCHS, self.lr)

        train_dice = [0.] * len(self.cfg.DATA_LOADER.LABEL_INDEX)
        train_total = 0

        for index, (images, masks) in enumerate(self.train_loader):
            if self.cfg.ENVIRONMENT.CUDA:
                images, masks = images.cuda(), masks.cuda()

            self.optimizer.zero_grad()
            if self.cfg.FINE_MODEL.DEEP_SUPERVISION:
                output_seg_list = self.model(images)
                seg_loss = 0
                loss_func = SegLoss(loss_func=self.cfg.TRAINING.LOSS, activation=self.cfg.TRAINING.ACTIVATION)
                for seg in output_seg_list:
                    seg_loss += loss_func(seg, masks, is_average=True)
                seg_loss /= len(output_seg_list)
                output_seg = output_seg_list[0]
            else:
                output_seg = self.model(images)
                loss_func = SegLoss(loss_func=self.cfg.TRAINING.LOSS, activation=self.cfg.TRAINING.ACTIVATION)
                seg_loss = loss_func(output_seg, masks, is_average=True)

            if self.is_apex_train:
                with amp.scale_loss(seg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                seg_loss.backward()
            self.optimizer.step()

            dice_metric_func = DiceMetric()
            dice_output = dice_metric_func(output_seg, masks, activation=self.cfg.TRAINING.ACTIVATION,
                                           is_average=False)
            if self.is_distributed_train:
                dice_output = self._reduce_tensor(dice_output.data)
            for i, dice_tmp in enumerate(dice_output):
                train_dice[i] += float(dice_tmp.item())
            train_total += len(images)

            if self.is_distributed_train:
                seg_loss = self._reduce_tensor(seg_loss.data)

            if self.is_print_out:
                if index > 0 and index % self.cfg.TRAINING.SAVER.SAVER_FREQUENCY == 0:
                    self.logger.info('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch, self.cfg.TRAINING.SCHEDULER.TOTAL_EPOCHS,
                        index * len(images), len(self.train_loader),
                        100. * index / len(self.train_loader)))
                    self.logger.info('SegLoss:{:.6f}'.format(seg_loss.item()))

                    for i, dice_label in enumerate(train_dice):
                        dice_ind = dice_label / train_total
                        self.logger.info('{} Dice:{:.6f}'.format(self.cfg.DATA_LOADER.LABEL_NAME[i], dice_ind))

        if self.is_print_out:
            self.train_writer.add_scalar('Train/SegLoss', seg_loss.item(), epoch)

            for i, dice_label in enumerate(train_dice):
                dice_ind = dice_label / train_total
                self.train_writer.add_scalars('Train/Dice',
                                              {self.cfg.DATA_LOADER.LABEL_NAME[i]: dice_ind}, epoch)

    def _validate(self, epoch):
        self.model.eval()

        val_dice = [0.] * len(self.cfg.DATA_LOADER.LABEL_INDEX)
        val_total = 0
        val_loss = 0
        if self.cfg.TRAINING.METRIC == 'diceAndHausdorff':
            is_hausdorff = True
            val_hausdorff = [0.] * len(self.cfg.DATA_LOADER.LABEL_INDEX)
        else:
            is_hausdorff = False

        for images, masks in self.val_loader:
            if self.cfg.ENVIRONMENT.CUDA:
                images, masks = images.cuda(), masks.cuda()

            with torch.no_grad():
                if self.cfg.FINE_MODEL.DEEP_SUPERVISION:
                    output_seg_list = self.model(images)
                    output_seg = output_seg_list[0]
                else:
                    output_seg = self.model(images)

            loss_func = SegLoss(loss_func='dice', activation=self.cfg.TRAINING.ACTIVATION)
            seg_loss = loss_func(output_seg, masks, is_average=False)

            if self.is_distributed_train:
                seg_loss = self._reduce_tensor(seg_loss.data)
            val_loss += seg_loss.item()

            if is_hausdorff:
                masks = masks.cpu().numpy()
                output_seg = output_seg.cpu().numpy()
                batch_num = masks.shape[0]
                dice_output = []
                hausdorff_output = []
                for i in range(batch_num):
                    area_dice, surface_dice = compute_flare_metric(masks[i], output_seg[i], [1.0, 1.0, 1.0])
                    dice_output.append(area_dice)
                    hausdorff_output.append(surface_dice)
                dice_output = np.array(dice_output).sum(0)
                hausdorff_output = np.array(hausdorff_output).sum(0)
                dice_output = torch.from_numpy(dice_output).float().to(self.device)
                hausdorff_output = torch.from_numpy(hausdorff_output).float().to(self.device)
            else:
                dice_metric_func = DiceMetric()
                dice_output = dice_metric_func(output_seg, masks,
                                               activation=self.cfg.TRAINING.ACTIVATION, is_average=False)

            if self.is_distributed_train:
                dice_output = self._reduce_tensor(dice_output.data)
                if is_hausdorff:
                    hausdorff_output = self._reduce_tensor(hausdorff_output.data)
            for i, dice_tmp in enumerate(dice_output):
                val_dice[i] += float(dice_tmp.item())
                if is_hausdorff:
                    val_hausdorff[i] += float(hausdorff_output[i].item())
            val_total += len(images)

        val_loss /= val_total
        total_dice = 0
        if self.is_print_out:
            self.logger.info('Loss of validation is {}'.format(val_loss))
            self.val_writer.add_scalar('Val/Loss', val_loss, epoch)

            for idx, _ in enumerate(val_dice):
                val_dice[idx] /= val_total
                self.logger.info('{} Dice:{:.6f}'.format(self.cfg.DATA_LOADER.LABEL_NAME[idx], val_dice[idx]))
                self.val_writer.add_scalars('Val/Dice',
                                            {self.cfg.DATA_LOADER.LABEL_NAME[idx]: val_dice[idx]}, epoch)
                total_dice += val_dice[idx]
                if is_hausdorff:
                    val_hausdorff[idx] /= val_total
                    self.logger.info('{} Hausdorff:{:.6f}'.format(self.cfg.DATA_LOADER.LABEL_NAME[idx],
                                                                  val_hausdorff[idx]))
                    self.val_writer.add_scalars('Val/Hausdorff',
                                                {self.cfg.DATA_LOADER.LABEL_NAME[idx]: val_hausdorff[idx]}, epoch)
                    total_dice += val_hausdorff[idx]
            total_dice /= len(val_dice)
            if is_hausdorff:
                total_dice /= 2

        return total_dice

    def do_test(self, fold=1):
        self.logger.info("start test {}th fold...".format(fold))

        self.coarse_model.eval()
        self.fine_model.eval()
        self._set_requires_grad(self.coarse_model, False)
        self._set_requires_grad(self.fine_model, False)

        test_dataset = SegDataSet(self.cfg, 'test')
        test_dataloader = DataLoaderX(
            dataset=test_dataset,
            batch_size=self.cfg.TESTING.BATCH_SIZE,
            num_workers=self.cfg.TESTING.NUM_WORKER,
            shuffle=False,
            drop_last=False,
            collate_fn=test_collate_fn,
            pin_memory=True)

        out_coarse_mask_dir = os.path.join(self.test_save_dir, 'coarse_mask')
        out_fine_mask_dir = os.path.join(self.test_save_dir, 'fine_mask')
        if not os.path.exists(out_coarse_mask_dir):
            os.makedirs(out_coarse_mask_dir)
        if not os.path.exists(out_fine_mask_dir):
            os.makedirs(out_fine_mask_dir)

        out_ind_csv_path = os.path.join(self.test_save_dir, 'ind_seg_result.csv')
        ind_content = ['series_uid', 'z_spacing']
        object_metric = []
        for object_name in self.cfg.DATA_LOADER.LABEL_NAME:
            object_metric.extend([object_name+'_DSC', object_name+'_NSC'])
        ind_content.extend(object_metric)
        ind_content.extend(['Average_DSC', 'Average_NSC', 'Data_loader_time', 'Coarse_infer_time',
                            'Coarse_postprocess_time', 'Fine_infer_time', 'Fine_postprocess_time',
                            'Time_usage', 'Memory_usage', 'Time_score', 'Memory_score'])
        write_csv(out_ind_csv_path, ind_content, mul=False, mod='w')
        out_total_csv_path = os.path.join(self.test_save_dir, 'total_seg_result.csv')
        total_content = copy.deepcopy(object_metric)
        total_content.extend(['Average_DSC', 'Average_NSC', 'Average_data_loader_time', 'Average_coarse_infer_time',
                              'Average_coarse_postprocess_time', 'Average_fine_infer_time',
                              'Average_fine_postprocess_time', 'Average_time_usage', 'Time_score'])
        write_csv(out_total_csv_path, total_content, mul=False, mod='w')

        area_dice = [0.] * len(self.cfg.DATA_LOADER.LABEL_INDEX)
        surface_dice = [0.] * len(self.cfg.DATA_LOADER.LABEL_INDEX)
        num_class = len(self.cfg.DATA_LOADER.LABEL_INDEX)
        is_print_metric = False

        self.logger.info('Starting test')
        self.logger.info('test samples: {}'.format(len(test_dataloader)))

        time_consume = {'data_loader': [], 'coarse_infer': [], 'coarse_postprocess': [],
                        'fine_infer': [], 'fine_postprocess': []}
        torch.cuda.synchronize()
        t_start = time.time()
        t_dataloader_start = time.time()
        for batch_idx, data_dict in enumerate(test_dataset):
            self.logger.info('process: {}/{}'.format(batch_idx, len(test_dataset)))

            # data_dict = data_dict[0]
            series_id = data_dict['series_id']
            raw_image = data_dict['image']
            raw_mask = data_dict['raw_mask']
            raw_spacing = data_dict['raw_spacing']
            image_direction = data_dict['image_direction']
            coarse_image = data_dict['coarse_input_image']
            coarse_zoom_factor = data_dict['coarse_zoom_factor']

            if raw_mask is not None:
                is_print_metric = True

            # segmentation in coarse resolution.
            self.logger.info('coarse segmentation start...')
            coarse_image = torch.from_numpy(coarse_image[np.newaxis, np.newaxis]).float()
            if self.cfg.ENVIRONMENT.CUDA:
                coarse_image = coarse_image.half() if self.cfg.TESTING.IS_FP16 else coarse_image
                coarse_image = coarse_image.cuda()

            torch.cuda.synchronize()
            t_dataloader_end = time.time()
            time_consume['data_loader'].append(t_dataloader_end-t_dataloader_start)

            with torch.no_grad():
                coarse_image = self.coarse_model(coarse_image)

            coarse_image = coarse_image.cpu().float()
            torch.cuda.empty_cache()
            pred_coarse_mask = coarse_image
            pred_coarse_mask = F.sigmoid(pred_coarse_mask)
            pred_coarse_mask = torch.where(pred_coarse_mask >= 0.5, torch.tensor(1), torch.tensor(0))
            pred_coarse_mask = pred_coarse_mask.numpy().squeeze(axis=0).astype(np.uint8)

            torch.cuda.synchronize()
            t_coarse_infer_end = time.time()
            time_consume['coarse_infer'].append(t_coarse_infer_end-t_dataloader_end)

            if self.cfg.TESTING.IS_POST_PROCESS:
                if self.cfg.COARSE_MODEL.NUM_CLASSES == 1:
                    label_num = 0
                    for i in self.cfg.DATA_LOADER.LABEL_NUM:
                        label_num += i
                    label_num = [label_num]
                else:
                    label_num = self.cfg.DATA_LOADER.LABEL_NUM
                coarse_spacing = [raw_spacing[i]*coarse_zoom_factor[i] for i in range(3)]
                area_least = 1000 / coarse_spacing[0] / coarse_spacing[1] / coarse_spacing[2]
                out_coarse_mask = extract_topk_largest_candidates(pred_coarse_mask, label_num, area_least)
            else:
                coarse_image_shape = pred_coarse_mask.shape
                out_coarse_mask = np.zeros(coarse_image_shape[1:])
                for i in range(coarse_image_shape[0]):
                    out_coarse_mask[pred_coarse_mask[i] != 0] = i+1

            # if self.cfg.TESTING.IS_SAVE_MASK:
            #     mask_path = os.path.join(out_coarse_mask_dir, series_id + ".nii.gz")
            #     save_ct_from_npy(out_coarse_mask, mask_path, spacing=[1.0, 1.0, 1.0], use_compression=True)
            #     self.logger.info('save coarse mask complete!')

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

            torch.cuda.synchronize()
            t_coarse_postprocess_end = time.time()
            time_consume['coarse_postprocess'].append(t_coarse_postprocess_end - t_coarse_infer_end)

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
            torch.cuda.synchronize()
            t_fine_infer_end = time.time()
            time_consume['fine_infer'].append(t_fine_infer_end-t_coarse_postprocess_end)

            if not self.cfg.FINE_MODEL.IS_POSTPROCESS:
                fine_mask = []
                for i in range(len(predict_fine_mask)):
                    mask, _ = ScipyResample.resample_to_size(predict_fine_mask[i], crop_image_size,
                                                             order=self.cfg.TESTING.OUT_RESAMPLE_MODE)
                    fine_mask.append(mask)
                predict_fine_mask = np.stack(fine_mask, axis=0)
                predict_fine_mask = np.where(predict_fine_mask >= 0.5, 1, 0)

            if self.cfg.TESTING.IS_POST_PROCESS:
                fine_spacing = [raw_spacing[i] * fine_zoom_factor[i] for i in range(3)]
                area_least = 2000 / fine_spacing[0] / fine_spacing[1] / fine_spacing[2]
                predict_fine_mask = extract_topk_largest_candidates(predict_fine_mask,
                                                                    self.cfg.DATA_LOADER.LABEL_NUM, area_least)
            else:
                t_mask = np.zeros(crop_image_size, np.uint8)
                for i in range(num_class):
                    t_mask[predict_fine_mask[i] != 0] = i + 1
                predict_fine_mask = t_mask

            out_mask = np.zeros(raw_image_shape, np.uint8)
            out_mask[crop_fine_bbox[0]:crop_fine_bbox[1],
                     crop_fine_bbox[2]:crop_fine_bbox[3],
                     crop_fine_bbox[4]:crop_fine_bbox[5]] = predict_fine_mask

            out_content = [series_id, raw_spacing[0]]
            if raw_mask is not None:
                predict_mask = np.zeros([num_class, raw_image_shape[0],
                                         raw_image_shape[1], raw_image_shape[2]], np.uint8)
                for i in range(num_class):
                    predict_mask[i] = out_mask == i + 1
                t_area_dice, t_surface_dice = compute_flare_metric(raw_mask, predict_mask, raw_spacing)
                total_area_dice = 0
                total_surface_dice = 0
                for i in range(num_class):
                    area_dice[i] += t_area_dice[i]
                    surface_dice[i] += t_surface_dice[i]
                    out_content.append(t_area_dice[i])
                    out_content.append(t_surface_dice[i])
                    total_area_dice += t_area_dice[i]
                    total_surface_dice += t_surface_dice[i]
                    self.logger.info('{} DSC: {}, NSC: {}'.format(self.cfg.DATA_LOADER.LABEL_NAME[i],
                                                                  t_area_dice[i], t_surface_dice[i]))
                self.logger.info('Average DSC: {}, Average NSC: {}'.format(total_area_dice / num_class,
                                                                           total_surface_dice / num_class))
                out_content.extend([total_area_dice / num_class, total_surface_dice / num_class])
                torch.cuda.synchronize()
                pipeline_end = time.time()
                pipeline_time_usage = pipeline_end - t_dataloader_start
                time_metric = (100 - pipeline_time_usage) * 1.0 / 100
                time_consume['fine_postprocess'].append(pipeline_end-t_fine_infer_end)
                gpu_memory_max, memory_metric_min = self.get_gpu_memory_usage(
                    self.cfg.ENVIRONMENT.MONITOR_TIME_INTERVAL, False, gpu_index=0)

                out_content.extend([time_consume['data_loader'][-1], time_consume['coarse_infer'][-1],
                                    time_consume['coarse_postprocess'][-1], time_consume['fine_infer'][-1],
                                    time_consume['fine_postprocess'][-1], pipeline_time_usage, gpu_memory_max,
                                    time_metric, memory_metric_min])
                write_csv(out_ind_csv_path, out_content, mul=False, mod='a+')

            if self.cfg.TESTING.IS_SAVE_MASK:
                mask_path = os.path.join(out_fine_mask_dir, series_id + ".nii.gz")
                if raw_spacing[0] != raw_spacing[1]:
                    raw_spacing = [raw_spacing[2], raw_spacing[1], raw_spacing[0]]
                out_mask = change_axes_of_image(out_mask, image_direction)
                save_ct_from_npy(out_mask, mask_path, spacing=raw_spacing, use_compression=True)
                self.logger.info('save fine mask complete!')

            t_dataloader_start = time.time()

        torch.cuda.synchronize()
        t_end = time.time()
        average_time_usage = (t_end - t_start) * 1.0 / len(test_dataset)
        time_score = (100-average_time_usage) * 1.0 / 100
        self.logger.info("Average time usage: {} s".format(average_time_usage))
        self.logger.info("Normalized time coefficient: {}".format(time_score))
        average_data_loader_time = np.mean(np.array(time_consume['data_loader']))
        average_coarse_infer = np.mean(np.array(time_consume['coarse_infer']))
        average_coarse_postprocess = np.mean(np.array(time_consume['coarse_postprocess']))
        average_fine_infer = np.mean(np.array(time_consume['fine_infer']))
        average_fine_postprocess = np.mean(np.array(time_consume['fine_postprocess']))
        self.logger.info('Average data loader time: {}'.format(average_data_loader_time))
        self.logger.info('Average coarse infer time: {}'.format(average_coarse_infer))
        self.logger.info('Average coarse postprocess time: {}'.format(average_coarse_postprocess))
        self.logger.info('Average fine infer time: {}'.format(average_fine_infer))
        self.logger.info('Average fine postprocess time: {}'.format(average_fine_postprocess))

        if is_print_metric:
            total_area_dice = 0
            total_surface_dice = 0
            for idx, _ in enumerate(area_dice):
                area_dice[idx] /= len(test_dataset)
                surface_dice[idx] /= len(test_dataset)
                self.logger.info('Average {} DSC:{:.6f}'.format(self.cfg.DATA_LOADER.LABEL_NAME[idx], area_dice[idx]))
                self.logger.info('Average {} NSC:{:.6f}'.format(self.cfg.DATA_LOADER.LABEL_NAME[idx], surface_dice[idx]))
                total_area_dice += area_dice[idx]
                total_surface_dice += surface_dice[idx]
            total_area_dice /= len(area_dice)
            total_surface_dice /= len(surface_dice)
            out_content = []
            for i in range(4):
                out_content.append(area_dice[i])
                out_content.append(surface_dice[i])
            out_content.extend([total_area_dice, total_surface_dice])
            out_content.extend([average_data_loader_time, average_coarse_infer, average_coarse_postprocess,
                                average_fine_infer, average_fine_postprocess, average_time_usage, time_score])

            write_csv(out_total_csv_path, out_content, mul=False, mod='a+')
            self.logger.info('Total average DSC: {:.6f}'.format(total_area_dice))
            self.logger.info('Total average NSC: {:.6f}'.format(total_surface_dice))

        self.logger.info("Test {}th fold finish!".format(fold))