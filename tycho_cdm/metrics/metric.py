import torch
import numpy as np
import pandas as pd
from tycho_cdm.utils.post_process import xywh2xyxy
import os


def conver_pd_tensor(data):
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
    my_tensor = torch.tensor(my_array, dtype=torch.float16)
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
    device = 'cpu'

    true_bbox_coord = box_coord(np.array(true_bbox))
    pred_bbox_coord = box_coord(np.array(pred_bbox))

    true_bbox_tensor = conver_pd_tensor(true_bbox_coord)
    pred_bbox_tensor = conver_pd_tensor(pred_bbox_coord)

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

    true_bbox_tensor = conver_pd_tensor(true_bbox_coord)
    pred_bbox_tensor = conver_pd_tensor(pred_bbox_coord)

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

# def caculate_single_bbox_iou(bbox1, bbox2):
#     bbox1_x_min = min(bbox1[0], bbox1[2])
#     bbox1_x_max = max(bbox1[0], bbox1[2])
#     bbox1_y_min = min(bbox1[1], bbox1[3])
#     bbox1_y_max = max(bbox1[1], bbox1[3])
#     bbox2_x_min = min(bbox2[0], bbox2[2])
#     bbox2_x_max = max(bbox2[0], bbox2[2])
#     bbox2_y_min = min(bbox2[1], bbox2[3])
#     bbox2_y_max = max(bbox2[1], bbox2[3])
#     box1 = (bbox1_x_min, bbox1_y_min, bbox1_x_max, bbox1_y_max)
#     box2 = (bbox2_x_min, bbox2_y_min, bbox2_x_max, bbox2_y_max)
#     bxmin = max(box1[0], box2[0])
#     bymin = max(box1[1], box2[1])
#     bxmax = min(box1[2], box2[2])
#     bymax = min(box1[3], box2[3])
#     bwidth = bxmax - bxmin
#     bhight = bymax - bymin
#     inter = bwidth * bhight
#     union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
#     return inter / union
#
#
# def read_single_boxes(predict_bbox, gt_bbox):
#     print('caculating the statices information.....')
#     TP, FP, FN = 0, 0, 0
#     predict_bboxes = xywh2xyxy(predict_bbox)
#     gt_bboxes = xywh2xyxy(gt_bbox)
#     for predict_bbox in predict_bboxes:
#         fn_status = 1
#         for gt_bbox in gt_bboxes:
#             iou = caculate_single_bbox_iou(predict_bbox, gt_bbox)
#             if iou > 0.5:
#                 TP = TP + 1
#                 fn_status = 0
#                 break
#         FN = FN + fn_status
#
#     for gt_bbox in gt_bboxes:
#         fp_status = 1
#         for predict_bbox in predict_bboxes:
#             iou = caculate_single_bbox_iou(predict_bbox, gt_bbox)
#             if iou > 0.5:
#                 fp_status = 0
#                 break
#         FP = FP + fp_status
#     return TP, FP, FN
#
#
# def read_boxes(predicted_boxes_for_image, true_boxes_for_image):
#     TP, FP, FN = 0, 0, 0
#     for index, pre_bbox in enumerate(predicted_boxes_for_image):
#         true_bbox = true_boxes_for_image[index]
#         temtp, temfp, temfn = read_single_boxes(pre_bbox, true_bbox)
#         TP = TP + temtp
#         FP = FP + temfp
#         FN = FN + temfn
#     return TP, FP, FN

