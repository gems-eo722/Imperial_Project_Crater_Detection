import os
import shutil

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmyolo.utils import register_all_modules

config_file = 'model/config/yolov8_l.py'
checkpoint_file = 'model/weight/bbox_mAP_epoch_550.pth'

list = ['A416', 'B416', 'C416', 'D416']
path = 'Data'
register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')

import numpy as np
from numpy import array


def xyxy2xywh(bbox, image_shape):
    new_bbox = np.zeros_like(bbox)
    new_bbox[:, 2] = np.absolute(bbox[:, 0] - bbox[:, 2]) / image_shape[1]
    new_bbox[:, 3] = np.absolute(bbox[:, 1] - bbox[:, 3]) / image_shape[0]
    new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / (2 * image_shape[1])
    new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / (2 * image_shape[0])
    return new_bbox


def box_area(boxes: array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1: array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes: array, scores: array, iou_threshold: float):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)  # Tensor
    bbox = boxes[keep]
    return bbox


def NMS(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[temp]


new_image_path = 'moon_data/moon_yolo_v3/images'
new_label_path = 'moon_data/moon_yolo_v3/labels'

for fold in list:
    data_path = os.path.join(path, fold)
    file_list = os.listdir(data_path)
    difference = []
    for file in file_list:
        if '.png' in file:
            try:
                image_name = file
                bbox_name = file.replace('.png', '.txt')
                bbox_path = os.path.join(data_path, bbox_name)
                ground_bbox = np.loadtxt(bbox_path, delimiter=',')
                ground_bbox[:, 4] = 1

                image_path = os.path.join(data_path, image_name)
                image = mmcv.imread(image_path, channel_order='rgb')

                result = inference_detector(model, image)
                score_result = result.pred_instances['scores']
                bbox_result = result.pred_instances['bboxes']

                index_score = score_result > 0.5
                score = score_result[index_score].detach().cpu().numpy()
                bbox = bbox_result[index_score].detach().cpu().numpy()
                weight = np.absolute(bbox[:, 0] - bbox[:, 2])
                height = np.absolute(bbox[:, 1] - bbox[:, 3])

                size = np.sqrt(weight * height)
                index_size = size < 24
                bbox = bbox[index_size]
                score = score[index_size]
                bbox = np.concatenate((bbox, score.reshape((len(score), 1))), axis=1)
                new_bbox = np.concatenate((ground_bbox, bbox), axis=0)
                new_bbox = numpy_nms(new_bbox[:, :4], new_bbox[:, 4], 0.5)

                new_bbox = xyxy2xywh(new_bbox, (416, 416))
                bbox = np.concatenate((np.zeros((len(new_bbox), 1)), new_bbox), axis=1)
                difference.append(len(bbox)-len(ground_bbox))

                np.savetxt(os.path.join(new_label_path, fold + '_' + bbox_name), bbox, delimiter=' ')
                shutil.copy(image_path, os.path.join(new_image_path, fold + '_' + image_name))
            except:
                pass