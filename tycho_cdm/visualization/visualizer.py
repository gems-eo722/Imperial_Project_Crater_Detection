import os
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

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

    labels = (np.loadtxt(label_path, delimiter=',')) if label_path is not None else None

    input_image = cv2.imread(image_path)
    image_name = Path(image_path).name.__str__()

    if len(bounding_boxes.shape) == 1:
        bounding_boxes = np.array([bounding_boxes])
    draw_bounding_boxes(input_image, bounding_boxes, confidences, (0, 0, 255),
                        draw_confidence=max(input_image.shape) <= int(416 * 1.5))

    if labels is not None:
        if len(labels.shape) == 1:
            labels = np.array([labels])
        height = input_image.shape[0]
        width = input_image.shape[1]
        labels[:, 0], labels[:, 2] = labels[:, 0] * width, labels[:, 2] * width
        labels[:, 1], labels[:, 3] = labels[:, 1] * height, labels[:, 3] * height
        labels = labels.astype(int)
        labels = xywh2xyxy(labels)
        draw_bounding_boxes(input_image, labels, confidences, (255, 0, 0), draw_confidence=False)

    cv2.imwrite(os.path.join(output_path, image_name), input_image)


def draw_bounding_boxes(image: np.ndarray, bounding_boxes: [np.ndarray], confidences, color, draw_confidence=True) -> None:
    """
    Draws the given bounding boxes onto the given image

    :param image: An image as numpy array
    :param bounding_boxes: The bounding boxes to plot, as a list of numpy arrays (2-d)
    :param confidences: The confidences for each bounding box prediction as list of float
    :param color: The color to draw the bounding boxes with
    :param draw_confidence: Whether to draw the confidence labels above the boxes
    """
    for i, bbox in enumerate(bounding_boxes):
        # Bounding rect
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color)

        if draw_confidence:
            write_confidence_text(color, confidences[i], image, bbox[0], bbox[1])


def write_confidence_text(background_color, confidence, image, top_left_x, top_left_y) -> None:
    """
    Writes the given confidence as text on the image, at the given location
    :param background_color: The color of the text background
    :param confidence: The confidence value to write
    :param image: The image to write on
    :param top_left_x: X coordinate for text
    :param top_left_y: Y coordinate for text
    """
    text = f'crater {confidence:.2}'
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # Text background
    cv2.rectangle(image, (top_left_x, top_left_y - 20), (top_left_x + text_size[0], top_left_y),
                  background_color, -1)
    cv2.putText(image, text, (top_left_x, int(top_left_y - 5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=2)


def plot_distribution_graph(folder_path: str, file_name: str, bboxes: [np.ndarray], diameters=None, resolution=100) -> None:
    """
    Plots the size-frequency distribution for the image given by {folder_path}/{file_name}.png

    :param folder_path: Path to the image's folder
    :param file_name: The basename of the image (without file extension)
    :param bboxes: The bounding boxes of the image
    :param diameters: The diameters calculated using additional data; if None, the average of box width and box height are used
    :param resolution: The physical image resolution in metres per pixel, if given by the user; if None, a default of 100 m/px is used
    """
    widths = [box[2] for box in bboxes]
    heights = [box[3] for box in bboxes]

    sns.set()
    if diameters is None:
        # No metadata, calculate diameter as mean of width and height of bounding rect
        plot = sns.displot(
            ([(w + h) / 2000 for (w, h) in (zip(widths * resolution, heights * resolution))]),
            kind='hist', log_scale=10, aspect=2)

        plt.xscale('log')
        plt.yscale('log')
    else:
        plot = sns.displot(diameters, kind='hist', log_scale=10, aspect=2)

        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel("Crater Diameter (km)")
    plt.title("Size-Frequency Distribution of Crater Sizes")

    plot.savefig(os.path.join(folder_path, f'{file_name}.png'))
    plt.close(plot.fig)
