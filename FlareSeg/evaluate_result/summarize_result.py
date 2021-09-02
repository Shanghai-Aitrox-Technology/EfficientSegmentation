
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


if __name__ == '__main__':
    result_path = '/fileser/zhangfan/efficientSegmentation/FlareSeg/evaluate_result/output/'
    get_1fold_average_result(result_path)