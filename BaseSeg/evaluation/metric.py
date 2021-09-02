
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

from .surface_dice import compute_surface_distances, compute_surface_dice_at_tolerance, \
    compute_dice_coefficient


def compute_flare_metric(gt_data, seg_data, case_spacing):
    all_dice = []
    all_surface_dice = []
    gt_data = gt_data.astype(np.uint8)
    seg_data = seg_data.astype(np.uint8)
    for i in range(4):
        gt_mask, pred_mask = gt_data[i], seg_data[i]
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_mask) == 0 and np.sum(pred_mask) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            surface_distances = compute_surface_distances(gt_mask, pred_mask, case_spacing)
            DSC_i = compute_dice_coefficient(gt_mask, pred_mask)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
        all_dice.append(DSC_i)
        all_surface_dice.append(NSD_i)

    return all_dice, all_surface_dice


class DiceMetric(nn.Module):

    def __init__(self, dims=(2, 3, 4)):
        super(DiceMetric, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, activation='sigmoid', is_average=True):
        predict = predict.float()
        gt = gt.float()

        if activation == 'sigmoid':
            pred = F.sigmoid(predict)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
        elif activation == 'softmax':
            pred = F.softmax(predict, dim=1)

        intersection = torch.sum(pred * gt, dim=self.dims)
        union = torch.sum(pred, dim=self.dims) + torch.sum(gt, dim=self.dims)
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dice = dice.mean(0) if is_average else dice.sum(0)

        return dice


def compute_dice(predict, gt):
    predict = predict.astype(np.float)
    gt = gt.astype(np.float)
    intersection = np.sum(predict * gt)
    union = np.sum(predict + gt)
    dice = (2. * intersection + 1e-8) / (union + 1e-8)

    return dice


def compute_precision_recall_f1(predict, gt, num_class):
    tp, tp_fp, tp_fn = [0.] * num_class, [0.] * num_class, [0.] * num_class
    precision, recall, f1 = [0.] * num_class, [0.] * num_class, [0.] * num_class
    for label in range(num_class):
        t_labels = gt == label
        p_labels = predict == label
        tp[label] += np.sum(t_labels == (p_labels * 2 - 1))
        tp_fp[label] += np.sum(p_labels)
        tp_fn[label] += np.sum(t_labels)
        precision[label] = tp[label] / (tp_fp[label] + 1e-8)
        recall[label] = tp[label] / (tp_fn[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

    return precision, recall, f1


def compute_overlap_metrics(pred_npy, target_npy, metrics=None):
    """
    Compute the overlap metric between predict and ground truth mask.
    including jaccard, dice, volume_similarity, false_negative, false_positive
    """
    if metrics is None:
        metrics = ['dice']

    for metric in metrics:
        if metric not in {'jaccard', 'dice', 'volume_similarity', 'false_negative', 'false_positive'}:
            raise ValueError('Does not exist the {} metric'.format(metric))

    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(target, pred)

    overlap_results = dict()
    for metric in metrics:
        if metric == 'jaccard':
            overlap_results['jaccard'] = overlap_measures_filter.GetJaccardCoefficient()
        elif metric == 'dice':
            overlap_results['dice'] = overlap_measures_filter.GetDiceCoefficient()
        elif metric == 'volume_similarity':
            overlap_results['volume_similarity'] = overlap_measures_filter.GetVolumeSimilarity()
        elif metric == 'false_negative':
            overlap_results['false_negative'] = overlap_measures_filter.GetFalseNegativeError()
        elif metric == 'false_positive':
            overlap_results['false_positive'] = overlap_measures_filter.GetFalsePositiveError()

    return overlap_results


def compute_hausdorff_distance(pred_npy, target_npy):
    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(target, pred)

    return hausdorff_distance_filter.GetHausdorffDistance()


def compute_surface_distance_statistics(pred_npy, target_npy):
    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)

    # Symmetric surface distance measures
    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or
    # inside relationship, is irrelevant)
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(pred)
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
    reference_surface = sitk.LabelContour(target)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
    # segmentations, though in our case it is. More on this below.
    surface_distance_results = dict()
    surface_distance_results['mean_surface_distance'] = np.mean(all_surface_distances)
    surface_distance_results['median_surface_distance'] = np.median(all_surface_distances)
    surface_distance_results['std_surface_distance'] = np.std(all_surface_distances)
    surface_distance_results['max_surface_distance'] = np.max(all_surface_distances)

    return surface_distance_results