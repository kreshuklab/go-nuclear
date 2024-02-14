"""Compare two input images (segmentation and groundtruth) and compute the average precision (AP)."""

from pathlib import Path
from multiprocessing import Pool

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import contingency_table
from skimage.transform import resize

from runstardist.utils import load_dataset


def _relabel(input):
    # unique labels are original, relabelled flat array is reshaped into a relabelled volume
    unique_labels, relabelled_flat_array = np.unique(input, return_inverse=True)
    return unique_labels, relabelled_flat_array.reshape(input.shape)


def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    unique_labels_gt, gt = _relabel(gt)
    unique_labels_seg, seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix, unique_labels_gt, unique_labels_seg


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.
    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        self.iou_matrix, self.unique_labels_gt, self.unique_labels_seg = _iou_matrix(gt, seg)

        self.df_iou_matrix = pd.DataFrame(self.iou_matrix, index=self.unique_labels_gt, columns=self.unique_labels_seg)
        # Find argmax for each row:
        argmax = np.argmax(self.iou_matrix, axis=1)
        # Find which value in unique_labels_seg corresponds to the argmax and add it as a column to self.df_iou_matrix
        matching_labels = self.unique_labels_seg[argmax]
        self.df_iou_matrix['matching_labels'] = matching_labels
        # Find max in each row in self.iou_matrix and add it as a column to self.df_iou_matrix
        max_iou = np.max(self.iou_matrix, axis=1)
        self.df_iou_matrix['max_iou'] = max_iou

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn),
        }


class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self, iou=None):
        if iou is not None:
            self.iou_range = [iou]
        else:
            self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        if np.all(gt_seg == gt_seg.flat[0]):
            return 1.0

        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # get contingency table to save individual IoU scores to file
        df_iou_matrix = sm.df_iou_matrix
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return acc, self.iou_range, df_iou_matrix


def compute_and_save_ap_scores(lab, seg, path_dir_seg, path_file_score, path_file_iou_matrix, data_id, save=True):
    # Compute scores
    ap = AveragePrecision()
    scores, _, df_iou_matrix = ap(seg, lab)

    # Save scores
    if save:
        df_iou_matrix.to_csv(path_file_iou_matrix)
        dict_scores = {'dir': str(path_dir_seg), data_id: scores}
        with open(path_file_score, 'w') as f:
            json.dump(dict_scores, f)

    return scores


def compare_image_pair(segmentation, groundtruth, method='ap', **kwargs):
    if method == 'ap':
        # accuracy::list, iou_range::list/np.array, iou_matrix::pd.DataFrame
        return compute_ap(segmentation, groundtruth, kwargs.get('size_reference', 'groundtruth'))
    else:
        raise NotImplementedError("Only AP is implemented.")


def compute_ap(segmentation, groundtruth, size_reference=None):
    """Compute Average Precision (AP) score for a given segmentation and ground truth.
    Args:
        segmentation (ndarray): predicted segmentation
        groundtruth (ndarray): ground truth segmentation
        standard (str): 'segmentation' or 'groundtruth'. If None, resize to match ground truth
    Returns:
        accuracy (list): list of AP scores for IoU thresholds in the range (0.5, 0.95) with a step of 0.05
        iou_range (list): list of IoU thresholds
        iou_matrix (pd.DataFrame): contingency table
    """
    print("got size_reference: ", size_reference)
    if segmentation.shape != groundtruth.shape:
        print("Segmentation and ground truth must have the same shape.")
        if size_reference == 'segmentation':
            print(f"Resizing ground truth {groundtruth.shape} to match segmentation {segmentation.shape}.")
            groundtruth = resize(groundtruth, segmentation.shape, preserve_range=True, order=0, anti_aliasing=False)
        else:
            print(f"Resizing segmentation {segmentation.shape} to match ground truth {groundtruth.shape}.")
            segmentation = resize(segmentation, groundtruth.shape, preserve_range=True, order=0, anti_aliasing=False)
    else:
        print("Segmentation and ground truth have the same shape.")
    return AveragePrecision()(segmentation, groundtruth)


def get_volume(image, name=None):
    if isinstance(image, str) or isinstance(image, Path):
        image, _ = load_dataset(image, Path(image).suffix, dset_name=name)
        return image
    elif isinstance(image, np.ndarray):
        return image


def compare_image_pairs(list_segmentation, list_groundtruth, method='ap', name_segmentation=None, name_groundtruth=None, **kwargs):
    if len(list_segmentation) != len(list_groundtruth):
        raise ValueError("Segmentation and ground truth must have the same length.")
    list_scores, list_iou_matrix = [], []
    for segmentation, groundtruth in zip(tqdm(list_segmentation), list_groundtruth):
        segmentation = get_volume(segmentation, name_segmentation)
        groundtruth = get_volume(groundtruth, name_groundtruth)
        scores, iou_range, iou_matrix = compare_image_pair(segmentation, groundtruth, method=method, **kwargs)
        list_scores.append(scores)
        list_iou_matrix.append(iou_matrix)
    return list_scores, iou_range, list_iou_matrix


def apply_compare_image_pair(segmentation_groundtruth_pair):
    segmentation, groundtruth = segmentation_groundtruth_pair
    return compare_image_pair(segmentation, groundtruth, method='ap')


def concurrent_basic_compare_image_pairs(list_segmentation, list_groundtruth, method='ap', name_segmentation=None, name_groundtruth=None):
    if len(list_segmentation) != len(list_groundtruth):
        raise ValueError("Segmentation and ground truth must have the same length.")
    list_scores, list_iou_matrix = [], []

    list_segmentation = [get_volume(seg, name_segmentation) for seg in list_segmentation]
    list_groundtruth = [get_volume(gt, name_groundtruth) for gt in list_groundtruth]

    with Pool(len(list_segmentation)) as p:
        results = p.map(apply_compare_image_pair, zip(list_segmentation, list_groundtruth))

    for scores, iou_range, iou_matrix in results:
        list_scores.append(scores)
        list_iou_matrix.append(iou_matrix)

    return list_scores, iou_range, list_iou_matrix


def save_scores_json(list_scores, list_id, iou_range, path_file_score):
    dict_scores = {}
    for scores, data_id in zip(list_scores, list_id):
        dict_scores[data_id] = scores
    dict_scores['iou_range'] = iou_range.tolist()
    with open(path_file_score, 'w') as f:
        json.dump(dict_scores, f)


def save_scores_csv(list_scores, list_id, iou_range, path_file_score):
    df_scores = pd.DataFrame(list_scores, columns=iou_range)
    df_scores['id'] = list_id
    df_scores.set_index('id', inplace=True)
    df_scores.to_csv(path_file_score)


def save_iou_matrix(list_iou_matrix, list_id, path_dir_iou_matrix):
    path_dir_iou_matrix = Path(path_dir_iou_matrix)
    if not path_dir_iou_matrix.exists():
        print(f"Create directory {path_dir_iou_matrix}")
        path_dir_iou_matrix.mkdir(parents=True)
    for iou_matrix, data_id in zip(list_iou_matrix, list_id):
        path_file_iou_matrix = path_dir_iou_matrix / (data_id + '.csv')
        iou_matrix.to_csv(path_file_iou_matrix)
