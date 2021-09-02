import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.image_io import load_ct_info
from Common.file_utils import read_txt, write_csv
from BaseSeg.evaluation.metric import compute_flare_metric


def process_compute_metric(name, gt_dir, predict_dir, csv_path):
    print('process {} start...'.format(name))
    gt_mask_path = gt_dir + name + '.nii.gz'
    predict_mask_path = predict_dir + name + '.nii.gz'
    num_class = 4

    gt_dict = load_ct_info(gt_mask_path)
    predict_dict = load_ct_info(predict_mask_path)
    gt_mask = gt_dict['npy_image']
    predict_mask = predict_dict['npy_image']
    spacing = gt_dict['spacing']

    mask_shape = gt_mask.shape
    gt_mask_czyx = np.zeros([num_class, mask_shape[0], mask_shape[1], mask_shape[2]])
    predict_mask_czyx = np.zeros([num_class, mask_shape[0], mask_shape[1], mask_shape[2]])
    for i in range(4):
        gt_mask_czyx[i] = gt_mask == i+1
        predict_mask_czyx[i] = predict_mask == i+1
    area_dice, surface_dice = compute_flare_metric(gt_mask_czyx, predict_mask_czyx, spacing)
    out_content = [name, spacing[0]]
    total_area_dice = 0
    total_surface_dice = 0
    object_labels = ['Liver', 'Kidney', 'Spleen', 'Pancreas']
    for i in range(num_class):
        out_content.append(area_dice[i])
        out_content.append(surface_dice[i])
        total_area_dice += area_dice[i]
        total_surface_dice += surface_dice[i]
        print('{} DSC: {}, NSC: {}'.format(object_labels[i], area_dice[i], surface_dice[i]))
    out_content.extend([total_area_dice / num_class, total_surface_dice / num_class])
    write_csv(csv_path, out_content, mul=False, mod='a+')
    print('Average_DSC: {}, Average_NSC: {}'.format(total_area_dice / num_class, total_surface_dice / num_class))
    print('process {} finish!'.format(name))


if __name__ == '__main__':
    series_uid_path = '/ssd/zhangfan/SemanticSegmentation/dataset/flare_data/file_list/val_series_uids.txt'
    gt_mask_dir = '/fileser/zhangfan/DataSet/flare_val_mask/flare_labeled_masks/'
    predict_mask_dir = '/fileser/zhangfan/efficientSegmentation/outputs/'
    out_ind_csv_dir = './output/'
    if not os.path.exists(out_ind_csv_dir):
        os.makedirs(out_ind_csv_dir)
    out_ind_csv_path = out_ind_csv_dir + 'ind_seg_result.csv'

    ind_content = ['series_uid', 'z_spacing']
    labels = ['Liver', 'Kidney', 'Spleen', 'Pancreas']
    object_metric = []
    for object_name in labels:
        object_metric.extend([object_name + '_DSC', object_name + '_NSC'])
    ind_content.extend(object_metric)
    ind_content.extend(['Average_DSC', 'Average_NSC'])
    write_csv(out_ind_csv_path, ind_content, mul=False, mod='w')

    file_names = read_txt(series_uid_path)
    for file_name in tqdm(file_names):
        p1 = Process(target=process_compute_metric,
                     args=(file_name, gt_mask_dir, predict_mask_dir, out_ind_csv_path))
        p1.daemon = False
        p1.start()