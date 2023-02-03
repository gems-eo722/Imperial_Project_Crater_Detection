import numpy as np
import pandas as pd
import torch


def convert_pd_tensor(data):
    """
    Convert from dataframe to tensor

    Parameters
    ----------
    data: dataframe
    
    Returns
    --------
    my_tensor: tensor
    """
    my_array = np.array(data)
    my_tensor = torch.tensor(my_array)
    return my_tensor


def box_coord(box_metric):
    """
    Return the upper-left and lower-right coordinates in dataframe

    Parameters
    -----------
    box_metric (arraylike)
    
    Returns
    --------
    df: dataframe
    """
    x1 = box_metric[:, 0] - box_metric[:, 2] / 2
    y1 = box_metric[:, 1] - box_metric[:, 3] / 2
    x2 = box_metric[:, 0] + box_metric[:, 2] / 2
    y2 = box_metric[:, 1] + box_metric[:, 3] / 2
    df = pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return df


def box_iou(true_bbox, pred_bbox, eps=1e-7):
    """
    Parameters
    ----------
    true_bbox: dataframe
    pred_bbox: dataframe
    
    Returns
    -------
    iou: Tensor(M,N)
        The MxN matrix containing the pairwise IoU values 
        for every element in true_bbox and prediction_bbox)
    """
    true_bbox_coord = box_coord(np.array(true_bbox))
    pred_bbox_coord = box_coord(np.array(pred_bbox))

    true_bbox_tensor = convert_pd_tensor(true_bbox_coord)
    pred_bbox_tensor = convert_pd_tensor(pred_bbox_coord)

    (a1, a2), (b1, b2) = true_bbox_tensor.unsqueeze(1).chunk(2, 2), pred_bbox_tensor.unsqueeze(0).chunk(2, 2)

    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    iou = inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    pred_area = (b2 - b1).prod(2)
    truth_area = (a2 - a1).prod(2)

    return iou, pred_area, truth_area


def single_confusion_matrix(iou_matrix, threshold):
    """
    Return TP, FP, FN
    Parameters
    ----------
    iou_matrix: tensor
    threshold: float

    Returns
    -------
    TP: int
    FP: int
    FN: int
    """
    matrix = iou_matrix.numpy() > threshold
    TP = np.sum(matrix)
    FP = np.sum(np.sum(matrix, axis=1) == 0)
    FN = np.sum(np.sum(matrix, axis=0) == 0)
    return TP, FP, FN


def classification(true_bbox, pred_bbox, iou_thres):
    """
    Return information of three categories of boxes
    Parameters
    ----------
    ture_bbox: dataframe
    pred_bbox: dataframe

    Inputs are x, y, w, h of true boxes and prediction boxes

    Returns
    -------
    TP_true_box: tensor
    TP_pred_box: tensor
    FN_box: tensor
    FP_box: tensor
    """

    true_bbox_coord = box_coord(np.array(true_bbox))
    pred_bbox_coord = box_coord(np.array(pred_bbox))

    true_bbox_tensor = convert_pd_tensor(true_bbox_coord)
    pred_bbox_tensor = convert_pd_tensor(pred_bbox_coord)

    iou_thres = 0.5
    iou = box_iou(true_bbox_tensor, pred_bbox_tensor)[0]

    x = torch.where(iou > iou_thres)
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    m0, m1, _ = matches.transpose().astype(int)

    TP_true_box = true_bbox_tensor.index_select(0, torch.tensor(m0))
    TP_pred_box = pred_bbox_tensor.index_select(0, torch.tensor(m1))
    FN_box = 0
    FP_box = 0
    if len(m0) < len(true_bbox_tensor):
        FN_box = true_bbox_tensor.index_select(0, torch.tensor(
            np.array(list(set(range(len(true_bbox_tensor))) - set(m0)))))
    if len(m1) < len(pred_bbox_tensor):
        FP_box = pred_bbox_tensor.index_select(0, torch.tensor(
            np.array(list(set(range(len(pred_bbox_tensor))) - set(m1)))))

    return TP_true_box, TP_pred_box, FN_box, FP_box


def read_boxes(predicted_boxes_for_image, true_boxes_for_image):
    """
    :param predicted_boxes_for_image: List of size N, where N is the number of images. Each element is a 2-d array, containing the predicted bounding boxes for the imgae.
    :param true_boxes_for_image: List of size N, where N is the number of images. Each element is a 2-d array, containing the true bounding boxes for the imgae.
    :return:
    """
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(predicted_boxes_for_image)):
        true_boxes = true_boxes_for_image[i]
        if len(true_boxes.shape) == 1:
            true_boxes = np.array([true_boxes])

        pred_boxes = predicted_boxes_for_image[i]
        if len(pred_boxes.shape) == 1:
            pred_boxes = np.array([pred_boxes])

        # calculate metric for all data
        TP += single_confusion_matrix(box_iou(true_boxes, pred_boxes)[0], 0.5)[0]
        FP += single_confusion_matrix(box_iou(true_boxes, pred_boxes)[0], 0.5)[1]
        FN += single_confusion_matrix(box_iou(true_boxes, pred_boxes)[0], 0.5)[2]

    return TP, FP, FN

