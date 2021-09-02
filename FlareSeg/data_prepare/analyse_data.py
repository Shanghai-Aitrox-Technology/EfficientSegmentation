import os
import cc3d
from tqdm import tqdm
import numpy as np
from skimage import measure

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from Common.file_utils import read_txt
from Common.image_io import load_ct_info
from Common.mask_process import extract_bbox


def analyse_image(image_dir):
    image_size = []
    image_spacing = []
    image_direction = []

    filenames = os.listdir(image_dir)
    for idx in tqdm(filenames):
        image_path = image_dir + idx
        data_dict = load_ct_info(image_path)
        spacing = data_dict['spacing']
        direction = data_dict['direction']
        image_array = data_dict['npy_image']

        image_spacing.append(spacing)
        image_direction.append(direction)
        image_shape = image_array.shape
        image_shape = [image_shape[0] * spacing[0],
                       image_shape[1] * spacing[1],
                       image_shape[2] * spacing[2]]
        image_size.append(image_shape)

    labels = ['z', 'y', 'x']
    plot_properties(image_size, 'size', labels, bins=80)
    plot_properties(image_spacing, 'spacing', labels, bins=80)
    plot_properties(image_direction, 'direction', labels, bins=80)


def analyse_mask(image_dir, mask_dir, series_ids, suffix='.nii.gz'):
    organ_name_list = ['liver', 'spleen', 'pancreas', 'left_kidney', 'right_kidney']
    feature_name_list = ['area', 'size', 'centroid', 'intensity', 'scale', 'volume_ratio']

    mask_properties_dict = {'abdomen_size': []}
    for organ_name in organ_name_list:
        feature_dict = {}
        for feature_name in feature_name_list:
            feature_dict[feature_name] = []
        mask_properties_dict[organ_name] = feature_dict

    for idx, file_name in enumerate(series_ids):
        print('Processing number is {}/{}'.format(idx, len(series_ids)))

        image_path = os.path.join(image_dir, file_name) + '_0000' + suffix
        mask_path = os.path.join(mask_dir, file_name) + suffix
        if os.path.exists(image_path) and os.path.exists(mask_path):
            data_dict = load_ct_info(image_path)
            npy_image = data_dict['npy_image']
            spacing = data_dict['spacing']

            data_dict = load_ct_info(mask_path)
            npy_mask = data_dict['npy_image']

            features_dict = get_properties(npy_image, npy_mask, spacing)
            abdomen_size = features_dict['abdomen_size']
            size_features = features_dict['size']
            area_features = features_dict['area']
            centroid_features = features_dict['centroid']
            intensity_features = features_dict['intensity']
            scale_features = features_dict['scale']
            volume_ratio = features_dict['volume_ratio']

            centroid_features = np.array(centroid_features)
            start_centroid_z = np.min(centroid_features[:, 0])
            start_centroid_y = np.min(centroid_features[:, 1])
            start_centroid_x = np.min(centroid_features[:, 2])

            mask_properties_dict['abdomen_size'].append(abdomen_size)
            for i, organ_name in enumerate(organ_name_list):
                if i > len(size_features)-1:
                    break
                mask_properties_dict[organ_name]['area'].append(area_features[i])
                mask_properties_dict[organ_name]['size'].append(size_features[i])
                mask_properties_dict[organ_name]['scale'].append(scale_features[i])
                mask_properties_dict[organ_name]['intensity'].append(intensity_features[i])

                centroid_shift = [centroid_features[i, 0] - start_centroid_z,
                                  centroid_features[i, 1] - start_centroid_y,
                                  centroid_features[i, 2] - start_centroid_x]
                mask_properties_dict[organ_name]['centroid'].append(centroid_shift)
                mask_properties_dict[organ_name]['volume_ratio'].append(volume_ratio[i])
    plot_properties(mask_properties_dict['abdomen_size'], 'abdomen_size', ['z', 'y', 'x'], bins=80)
    for organ_name in organ_name_list:
        organ_features = mask_properties_dict[organ_name]
        area_features = organ_features['area']
        volume_ratio = organ_features['volume_ratio']
        size_features = organ_features['size']
        centroid_features = organ_features['centroid']
        intensity_features = organ_features['intensity']
        scale_features = organ_features['scale']
        plot_properties(area_features, organ_name, ['area'])
        plot_properties(volume_ratio, organ_name, ['volume_ratio'])
        plot_properties(size_features, organ_name, ['z_size', 'y_size', 'x_size'])
        plot_properties(centroid_features, organ_name, ['z_shift', 'y_shift', 'x_shift'])
        plot_properties(intensity_features, organ_name, ['max_intensity', 'mean_intensity', 'min_intensity'])
        plot_properties(scale_features, organ_name, ['major_axis_length', 'minor_axis_length', 'ratio'])


def get_properties(npy_image, npy_mask, spacing):
    # adjust mask label
    labeled_mask = npy_mask.copy()
    labeled_mask[labeled_mask == 2] = 0
    labeled_mask[labeled_mask == 3] = 2
    labeled_mask[labeled_mask == 4] = 3
    kidney_mask = npy_mask.copy()
    kidney_mask = np.where(kidney_mask == 2, 1, 0)
    keep_topk_largest_connected_object(kidney_mask, 2, 100, labeled_mask, [4, 5])

    mask_bbox = extract_bbox(npy_mask)
    mask_size = [(mask_bbox[0, 1]-mask_bbox[0, 0])*spacing[0],
                 (mask_bbox[1, 1]-mask_bbox[1, 0])*spacing[1],
                 (mask_bbox[2, 1]-mask_bbox[2, 0])*spacing[2]]
    mask_raw_size = [mask_bbox[0, 1]-mask_bbox[0, 0],
                     mask_bbox[1, 1]-mask_bbox[1, 0],
                     mask_bbox[2, 1]-mask_bbox[2, 0]]

    area_features = []
    volume_ratio = []
    size_features = []
    centroid_features = []
    intensity_features = []
    scale_features = []
    scale = spacing[0] * spacing[1] * spacing[2]
    region_props = measure.regionprops(labeled_mask, npy_image)
    for i in range(len(region_props)):
        area = region_props[i].area * scale
        area_features.append(area)

        volume_ratio.append(region_props[i].area*1.0/(mask_raw_size[0]*mask_raw_size[1]*mask_raw_size[2]))

        bbox = region_props[i].bbox
        size = [(bbox[3] - bbox[0]) * spacing[0],
                (bbox[4] - bbox[1]) * spacing[1],
                (bbox[5] - bbox[2]) * spacing[2]]
        size_features.append(size)

        centroid = region_props[i].centroid
        centroid = [centroid[0] * spacing[0],
                    centroid[1] * spacing[1],
                    centroid[2] * spacing[2]]
        centroid_features.append(centroid)

        max_intensity = region_props[i].max_intensity
        mean_intensity = region_props[i].mean_intensity
        min_intensity = region_props[i].min_intensity
        intensity_features.append([max_intensity, mean_intensity, min_intensity])

        major_axis_length = region_props[i].major_axis_length * scale
        minor_axis_length = region_props[i].minor_axis_length * scale
        scale_features.append([major_axis_length, minor_axis_length, major_axis_length / minor_axis_length])

    return {'area': area_features, 'size': size_features, 'centroid': centroid_features, 'volume_ratio': volume_ratio,
            'intensity': intensity_features, 'scale': scale_features, 'abdomen_size': mask_size}


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out, _ = cc3d.connected_components(npy_mask, return_N=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = np.sum(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label[i]


def plot_properties(properties, name, labels, bins=40):
    properties = np.array(properties)

    if len(labels) == 3:
        plt.figure()
        plt.subplot(131)
        plt.hist(properties[:, 0], bins=bins)
        plt.title('{} {}'.format(name, labels[0]))
        plt.subplot(132)
        plt.hist(properties[:, 1], bins=bins)
        plt.title('{} {}'.format(name, labels[1]))
        plt.subplot(133)
        plt.hist(properties[:, 2], bins=bins)
        plt.title('{} {}'.format(name, labels[2]))
        plt.show()

        print('{} mean is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.mean(properties[:, 0]),
                                                      labels[1], np.mean(properties[:, 1]),
                                                      labels[2], np.mean(properties[:, 2])))
        print('{} std is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.std(properties[:, 0]),
                                                     labels[1], np.std(properties[:, 1]),
                                                     labels[2], np.std(properties[:, 2])))
        print('{} max is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.max(properties[:, 0]),
                                                     labels[1], np.max(properties[:, 1]),
                                                     labels[2], np.max(properties[:, 2])))
        print('{} min is {}:{}, {}:{}, {}:{}'.format(name, labels[0], np.min(properties[:, 0]),
                                                     labels[1], np.min(properties[:, 1]),
                                                     labels[2], np.min(properties[:, 2])))
    elif len(labels) == 1:
        plt.figure()
        plt.hist(properties, bins=bins)
        plt.title('{} {}'.format(name, labels[0]))
        plt.show()
        print('{} mean is {}:{}'.format(name, labels[0], np.mean(properties)))
        print('{} std is {}:{}'.format(name, labels[0], np.std(properties)))
        print('{} max is {}:{}'.format(name, labels[0], np.max(properties)))
        print('{} min is {}:{}'.format(name, labels[0], np.min(properties)))


if __name__ == '__main__':
    # image_dir = "/ssd/zhangfan/SemanticSegmentation/dataset/flare_data/val_images/"
    # analyse_image(image_dir)
    image_dir = "/ssd/zhangfan/SemanticSegmentation/dataset/flare_data/train_images/"
    mask_dir = "/ssd/zhangfan/SemanticSegmentation/dataset/flare_data/train_mask/TrainingMask/"
    file_path = "/ssd/zhangfan/SemanticSegmentation/dataset/flare_data/file_list/train_series_uids.txt"
    series_ids = read_txt(file_path)
    analyse_mask(image_dir, mask_dir, series_ids, suffix='.nii.gz')