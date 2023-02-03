import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tycho_cdm.utils.post_process import xywh2xyxy


def visualize(image_path: str, bounding_boxes: [np.ndarray], confidences, output_path: str, label_path=None):
    """
    Creates images to visualize the location of predicted bounding boxes, and of the true bounding boxes,
    if available.

    :param image_path: Path to the input image
    :param bounding_boxes: List of numpy arrays containing in each array the x, y, w, h of a bounding box for the image
    :param confidences: Confidence of object detection model for this bounding
    :param output_path: Path to write visualization images to
    :param label_path: Optional, path to .csv file containing true bounding boxes for the input image
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    labels = (np.loadtxt(label_path,delimiter=',')) if label_path is not None else None

    input_image = cv2.imread(image_path)
    image_name = Path(image_path).name.__str__()

    draw_bounding_boxes(input_image, bounding_boxes, confidences, (0, 0, 255))

    if labels is not None:
        height = input_image.shape[0]
        width = input_image.shape[1]
        labels[:, 0], labels[:, 2] = labels[:, 0] * width, labels[:, 2] * width
        labels[:, 1], labels[:, 3] = labels[:, 1] * height, labels[:, 3] * height
        labels = labels.astype(int)
        labels = xywh2xyxy(labels)
        draw_bounding_boxes(input_image, labels, confidences, (255, 0, 0), draw_confidence=False)

    cv2.imwrite(os.path.join(output_path, image_name), input_image)


def draw_bounding_boxes(image: np.ndarray, bounding_boxes: [np.ndarray], confidences, color, draw_confidence=True):
    for i, bbox in enumerate(bounding_boxes):
        # top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_box_corners(bbox, image.shape[1], image.shape[0])

        # Bounding rect
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color)

        # if draw_confidence:
        #     write_confidence_text(color, confidences[i], image, bbox[0], bbox[1])


def write_confidence_text(color, confidence, image, top_left_x, top_left_y):
    text = f'crater {confidence:.2}'
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # Text background
    cv2.rectangle(image, (top_left_x, top_left_y - 20), (top_left_x + text_size[0], top_left_y),
                  color, -1)
    cv2.putText(image, text, (top_left_x, int(top_left_y - 5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=2)


def get_box_corners(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """
    :param bbox: Series containing x, y, width, height description of bounding box (where (x, y) is the box center)
    :param width: Image width in pixels
    :param height: Image height in pixels
    :return: Top-left and bottom-right points of bounding box as (x1, y2, x2, y2)
    """
    top_left_x = int(bbox[0] * width) - (int(bbox[2] * width) // 2)
    top_left_y = int(bbox[1] * height) - (int(bbox[3] * height) // 2)
    bottom_right_x = top_left_x + int(bbox[2] * width)
    bottom_right_y = top_left_y + int(bbox[3] * height)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y
