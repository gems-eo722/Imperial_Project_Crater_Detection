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
        draw.rectangle(
            ((int(bbox[0] * image.width), int(bbox[1] * image.height)),
             (int(bbox[2] * image.width), int(bbox[3] * image.height))),
            outline=color, width=2)
