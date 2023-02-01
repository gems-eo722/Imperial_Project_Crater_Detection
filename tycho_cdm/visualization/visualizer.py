import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw


def visualize(image_path: str, bounding_boxes: [np.ndarray], output_path: str, label_path=None):
    """
    Creates images to visualize the location of predicted bounding boxes, and of the true bounding boxes,
    if available.

    :param image_path: Path to the input image
    :param bounding_boxes: List of numpy arrays containing in each array the x, y, w, h of a bounding box for the image
    :param output_path: Path to write visualization images to
    :param label_path: Optional, path to .csv file containing true bounding boxes for the input image

    >>> df = pd.read_csv('../../example/Mars_THEMIS_Training/labels/aeolis_30_6.csv', header=None)
    >>> visualize('../../example/Mars_THEMIS_Training/images/aeolis_30_6.png', df, 'my_output')
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    input_image = Image.open(image_path)
    image_name = Path(image_path).name.__str__()

    rgba_image = Image.new("RGB", input_image.size)
    rgba_image.paste(input_image)

    labels = (pd.read_csv(label_path, header=None).to_numpy()) if label_path is not None else None

    visualized_image = rgba_image.copy()
    draw_bounding_boxes(visualized_image, bounding_boxes, (255, 0, 0))

    if labels is not None:
        draw_bounding_boxes(visualized_image, labels, (0, 0, 255))

    visualized_image.save(os.path.join(output_path, image_name), 'PNG')

    visualized_image.close()
    rgba_image.close()


def draw_bounding_boxes(image: Image, bounding_boxes: [np.ndarray], color):
    for bbox in bounding_boxes:
        draw = ImageDraw.Draw(image)
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_box_corners(bbox, image.width, image.height)
        draw.rectangle(
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y),
            outline=color, width=2)


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
