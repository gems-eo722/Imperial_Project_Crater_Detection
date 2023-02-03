import numpy as np


def xywh2xyxy(boxes):
    newboxes = np.zeros_like(boxes)
    newboxes[:, 0] = boxes[:, 0] - boxes[:, 2] // 2
    newboxes[:, 1] = boxes[:, 1] - boxes[:, 3] // 2
    newboxes[:, 2] = boxes[:, 0] + boxes[:, 2] // 2
    newboxes[:, 3] = boxes[:, 1] + boxes[:, 3] // 2
    return newboxes


def xyxy2xywh(boxes):
    newboxes = np.zeros_like(boxes)
    newboxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) // 2
    newboxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) // 2
    newboxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    newboxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return newboxes


def box_area(boxes: np.ndarray):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1: np.ndarray, box2: np.ndarray):
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


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
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

    return keep


def inference(img_orig, model):
    h, w = img_orig.shape[:-1]
    initial_subsize = subsize = 416
    all_boxes = []
    all_scores = []
    n = 1
    if max(h, w) <= int(416 * 1.5):
        output = model.single_inference(img_orig)
        bboxes = xywh2xyxy(output[0] * subsize)
        return bboxes.astype(np.int), output[1], output[2]
    else:
        while subsize < min(h, w):
            splitimgs, position = split(img_orig, subsize)
            image_boxes, _, image_boxes_scores = model.split_batch_inference_(splitimgs, 0.5)
            for i, img in enumerate(splitimgs):
                bboxes = image_boxes[i]
                bboxes[:, 0] += position[i, 0]
                bboxes[:, 2] += position[i, 0]
                bboxes[:, 1] += position[i, 1]
                bboxes[:, 3] += position[i, 1]
                all_boxes.append(bboxes)
                all_scores.append(image_boxes_scores[i])
            subsize = initial_subsize * pow(3, n)
            n += 1
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        if len(all_boxes) != len(all_scores):
            raise RuntimeError("Labels out of sync with boxes")
        indexes = numpy_nms(all_boxes, all_scores, 0.5)
        return all_boxes[indexes].astype(np.int), None, all_scores[indexes]


def split(img, subsize: int):
    position = []
    split_imgs = []
    gap = 0  # not needed anymore
    img_h, img_w = img.shape[:2]
    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize >= img_h:
            reachbottom = True
            top = max(img_h - subsize, 0)
        while not reachright:
            if left + subsize >= img_w:
                reachright = True
                left = max(img_w - subsize, 0)
            imgsplit = img[top:min(top + subsize, img_h), left:min(left + subsize, img_w)]
            if imgsplit.shape[:2] != (subsize, subsize):
                template = np.zeros((subsize, subsize, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template
            position.append([left, top])
            split_imgs.append(imgsplit)
            left += subsize - gap
        top += subsize - gap
    return split_imgs, np.asarray(position)
