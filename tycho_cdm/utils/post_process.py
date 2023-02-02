import numpy as np

import numpy as np


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


def inference(img, model):
    h, w = img.shape[:-2]
    subsize = 416
    all_boxes = []
    n = 1
    if max(h, w) <= int(416 * 1.5):
        outputs = model.batch_inference(img)
        boxes = outputs[0][1]
    else:
        while subsize < min(h, w):
            imgs, position = split(img, subsize)
            subsize = subsize * pow(3, n)
            outputs = model.batch_inference(imgs)
            for output in outputs:
                bboxes = output[1]
                bboxes[:, 0] += position[:, 0]
                bboxes[:, 2] += position[:, 0]
                bboxes[:, 1] += position[:, 1]
                bboxes[:, 3] += position[:, 1]
                all_boxes.append(bboxes)
            n += 1
        all_boxes = np.concatenate(all_boxes, axis=0)
        boxes = NMS(all_boxes, 0.5)
    return boxes


def split(img, subsize: int):
    position = []
    split_imgs = []
    gap = int(0.2 * subsize)
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
    return np.concatenate(split_imgs, axis=0), np.asarray(position)
