import numpy as np
import pandas as pd
from skimage.metrics import contingency_table


def _relabel(input):
    unique_labels, relabelled_flat_array = np.unique(input, return_inverse=True)
    return unique_labels, relabelled_flat_array.reshape(input.shape)


def _iou_matrix(gt, seg):
    unique_labels_gt, gt = _relabel(gt)
    unique_labels_seg, seg = _relabel(seg)

    n_inter = contingency_table(gt, seg).A

    n_gt = n_inter.sum(axis=1, keepdims=True)
    n_seg = n_inter.sum(axis=0, keepdims=True)

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
        argmax = np.argmax(self.iou_matrix, axis=1)
        matching_labels = self.unique_labels_seg[argmax]
        self.df_iou_matrix['matching_labels'] = matching_labels
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

        sm = SegmentationMetrics(gt_seg, input_seg)
        df_iou_matrix = sm.df_iou_matrix
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        return acc, self.iou_range, df_iou_matrix
