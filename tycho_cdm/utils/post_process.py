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


def NMS(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

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
    return temp


def inference(img_orig, model):
    h, w = img_orig.shape[:-1]
    initial_subsize = subsize = 416
    all_boxes = []
    all_scores = []
    n = 1
    if max(h, w) <= int(416 * 1.5):
        return model.single_inference(img_orig)
    else:
        while subsize < min(h, w):
            splitimgs, position = split(img_orig, subsize)
            for i, img in enumerate(splitimgs):
                output = model.single_inference(img)
                bboxes = output[0] * subsize
                bboxes = xywh2xyxy(bboxes)
                bboxes[:, 0] += position[i, 0]
                bboxes[:, 2] += position[i, 0]
                bboxes[:, 1] += position[i, 1]
                bboxes[:, 3] += position[i, 1]
                all_boxes.append(bboxes)
                all_scores.append(output[2])
            subsize = initial_subsize * pow(3, n)
            n += 1
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        if len(all_boxes) != len(all_scores):
            raise RuntimeError("Labels out of sync with boxes")
        indexes = NMS(all_boxes, all_scores, 0.5)
        return all_boxes[indexes].astype(np.int), None, all_scores[indexes]


def split(img, subsize: int):
    position = []
    split_imgs = []
    gap = 0 # not needed anymore
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
