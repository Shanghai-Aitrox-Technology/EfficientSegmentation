
import os
import numpy as np
import pandas as pd

from Common.file_utils import load_df


def get_1fold_average_result(result_path):
    file_name = 'ind_seg_result.csv'
    result_path += file_name
    data_df = load_df(result_path)
    column_header = ['Liver_DSC', 'Liver_NSC', 'Kidney_DSC', 'Kidney_NSC', 'Spleen_DSC', 'Spleen_NSC',
                     'Pancreas_DSC', 'Pancreas_NSC', 'Average_DSC', 'Average_NSC', 'Data_loader_time',
                     'Coarse_infer_time', 'Coarse_postprocess_time', 'Fine_infer_time', 'Time_usage',
                     'Memory_usage', 'Time_score', 'Memory_score']
    df_column_header = data_df.columns.values
    for name in column_header:
        if name in df_column_header:
            data = data_df[name].values
            print('{}: {}'.format(name, np.mean(data)))


def get_5fold_average_result(result_dir):
    result_folder = os.listdir(result_dir)
    all_result = []
    column_header = ['Liver_DSC', 'Liver_NSC', 'Kidney_DSC', 'Kidney_NSC', 'Spleen_DSC', 'Spleen_NSC',
                     'Pancreas_DSC', 'Pancreas_NSC', 'Average_DSC', 'Average_NSC']
    for fold in result_folder:
        result_path = result_dir + fold + '/ind_seg_result.csv'
        data_df = load_df(result_path)
        df_column_header = data_df.columns.values
        fold_result = []
        print('\n{}:'.format(fold))
        for name in column_header:
            if name in df_column_header:
                data = data_df[name].values
                fold_result.append(data)
                print('mean {}: {}'.format(name, np.mean(data)))
                print('std {}: {}'.format(name, np.std(data)))
        all_result.append(fold_result)
    all_result = np.array(all_result)
    print('\naverage of 5-fold cross validation: ')
    for idx, name in enumerate(column_header):
        print('mean {}: {}'.format(name, np.mean(all_result[:, idx])))
        print('std {}: {}'.format(name, np.std(all_result[:, idx])))


if __name__ == '__main__':
    # result_path = '/fileser/zhangfan/efficientSegmentation/FlareSeg/evaluate_result/output/'
    # get_1fold_average_result(result_path)
    result_dir = '/fileser/zhangfan/efficientSegmentation_refractor/FlareSeg/fine_efficient_seg/output_fold5_exp_0902/' \
                 'test/experiment_2_EfficientSegNet_ResBaseConvBlock_AnisotropicConvBlock_AnisotropicAvgPooling_size-' \
                 '192_channel-16_depth-4_loss-diceAndFocal_metric-dice/'
    get_5fold_average_result(result_dir)