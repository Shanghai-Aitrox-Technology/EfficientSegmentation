
import cc3d
import fastremap
import numpy as np
from typing import List
from skimage import measure
from skimage.morphology import label
import scipy.ndimage.morphology as morphology


def crop_image_according_to_mask(npy_image, npy_mask, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    bbox = extract_bbox(npy_mask)
    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([npy_image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    crop_image = npy_image[
                 extend_bbox[0, 0]:extend_bbox[0, 1],
                 extend_bbox[1, 0]:extend_bbox[1, 1],
                 extend_bbox[2, 0]:extend_bbox[2, 1]]
    crop_mask = npy_mask[
                extend_bbox[0, 0]:extend_bbox[0, 1],
                extend_bbox[1, 0]:extend_bbox[1, 1],
                extend_bbox[2, 0]:extend_bbox[2, 1]]

    return crop_image, crop_mask


def crop_image_according_to_bbox(npy_image, bbox, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    image_shape = npy_image.shape
    extend_bbox = [max(0, int(bbox[0]-margin[0])),
                   min(image_shape[0], int(bbox[1]+margin[0])),
                   max(0, int(bbox[2]-margin[1])),
                   min(image_shape[1], int(bbox[3]+margin[1])),
                   max(0, int(bbox[4]-margin[2])),
                   min(image_shape[2], int(bbox[5]+margin[2]))]
    crop_image = npy_image[extend_bbox[0]:extend_bbox[1],
                           extend_bbox[2]:extend_bbox[3],
                           extend_bbox[4]:extend_bbox[5]]

    return crop_image, extend_bbox


def convert_mask_2_one_hot(npy_mask, label=None):
    """Convert mask label into one hot coding."""
    if label is None:
        label = [1]

    npy_masks = []
    for i_label in range(1, np.max(np.array(label)) + 1):
        mask_i = (npy_mask == i_label)
        npy_masks.append(mask_i)

    npy_mask_czyx = np.stack(npy_masks, axis=0)
    npy_mask_czyx = npy_mask_czyx.astype(np.uint8)
    return npy_mask_czyx


def extract_bbox(npy_mask):
    ptc_ori = fastremap.point_cloud(npy_mask)
    ptc = np.vstack([ptc_ori[key] for key in ptc_ori.keys()])
    min_z, max_z = fastremap.minmax(ptc[:, 0])
    min_y, max_y = fastremap.minmax(ptc[:, 1])
    min_x, max_x = fastremap.minmax(ptc[:, 2])
    bbox = np.array([[min_z, max_z],
                     [min_y, max_y],
                     [min_x, max_x]])
    return bbox


def smooth_mask(npy_mask, out_mask, out_num_label=1, area_least=10, is_binary_close=False):
    if is_binary_close:
        struct = morphology.generate_binary_structure(3, 2)
        npy_mask = morphology.binary_closing(npy_mask, structure=struct, iterations=3)
    npy_mask = npy_mask.astype(np.uint8)
    remove_small_connected_object(npy_mask, area_least, out_mask, out_num_label)


def remove_small_connected_object(npy_mask, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        area = np.sum(extracted)
        if area > area_least:
            out_mask[labels_out == int(label)] = out_label


def extract_topk_largest_candidates(npy_mask: np.array, out_num_label: List, area_least: int) -> np.array:
    mask_shape = npy_mask.shape
    out_mask = np.zeros([mask_shape[1], mask_shape[2], mask_shape[3]], np.uint8)
    for i in range(mask_shape[0]):
        t_mask = npy_mask[i].copy()
        keep_topk_largest_connected_object(t_mask, out_num_label[i], area_least, out_mask, i+1)

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label


def extract_candidate_centroid(npy_mask, area_least, kth):
    npy_mask[npy_mask != 0] = 1
    npy_mask, num = label(npy_mask, neighbors=4, background=0, return_num=True)
    if num == 0:
        return []
    region_props = measure.regionprops(npy_mask)

    areas = {}
    centroids = []
    for i in range(num):
        t_area = region_props[i].area
        if t_area > area_least:
            areas[str(i)] = t_area
            centroids.append(region_props[i].centroid)
        else:
            centroids.append([0, 0, 0])

    if len(areas) == 0:
        return []

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    out_centroids = []
    for i in range(min(kth, len(candidates))):
        out_centroids.append(centroids[int(candidates[i][0])])

    return out_centroids

