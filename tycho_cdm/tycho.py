import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tycho_cdm.metrics import metric
from tycho_cdm.model.TychoCDM import TychoCDM
from tycho_cdm.utils.coordinate_conversions import get_lat_long_and_diameter
from tycho_cdm.utils.post_process import xyxy2xywh
from tycho_cdm.visualization import visualizer
from tycho_cdm.visualization.visualizer import plot_distribution_graph


def main():
    """
    Starts the program when using the command line.
    """
    parser = make_parser()
    input_folder, output_folder, planet_name = parse_arguments(parser)

    images_folder, labels_folder, metadata_folder = get_input_directories(input_folder)
    check_arguments(input_folder, output_folder, images_folder, planet_name)

    print("[TYCHO CDM] Parsed arguments successfully")

    print("[TYCHO CDM] Creating model...")
    model = TychoCDM(planet_name)
    print("[TYCHO CDM] Running inference on image batch...")
    results = model.batch_inference(images_folder)
    print("[TYCHO CDM] Inference complete, writing files to output")
    write_results(results, labels_folder, metadata_folder, output_folder)


def parse_arguments(parser: argparse.ArgumentParser):
    """
    Uses the given parser to parse the required command line arguments.
    :param parser:
    :return: The paths to the input folder and output folder, and the planet name
    """
    args = parser.parse_args()
    return args.input_folder, args.output_folder, args.planet_name


def check_arguments(input_folder: str, output_folder: str, images_folder: str, planet_name: str) -> None:
    """
    Checks that the input, output, and images folders, and planet name given by the user are valid.
    """

    if not os.path.isdir(input_folder):
        raise RuntimeError(f'Given path to input directory is invalid: {input_folder}')

    if not os.path.isdir(images_folder):
        raise RuntimeError('Input directory does not contain \'images\' subdirectory')
    if dir_is_empty(images_folder):
        raise RuntimeError('\'image\' subdirectory is empty, nothing to do')

    if (os.path.isdir(output_folder)) and not dir_is_empty(output_folder):
        raise RuntimeError('Please choose an output directory that is empty')

    if planet_name.lower() != 'moon' and planet_name.lower() != 'mars':
        raise RuntimeError('The planet name must be \'mars\' or \'moon\'')


def get_input_directories(input_folder) -> tuple[str, str | None, str | None]:
    """
    Returns the paths corresponding to the subdirectories in the input folder.

    :param input_folder: The input folder
    :return: input_folder_path/images, input_folder_path/labels (optional), and input_folder_path/data paths (optional)
    """
    labels_folder = check_path_is_directory(os.path.join(input_folder, 'labels'))
    metadata_folder = check_path_is_directory(os.path.join(input_folder, 'data'))
    images_folder = os.path.join(input_folder, 'images')
    return images_folder, labels_folder, metadata_folder


def get_output_directories(output_folder) -> tuple[str, str | None, str | None]:
    """
    Returns the paths corresponding to the subdirectories in the input folder.

    :param output_folder: The output folder
    :return: input_folder_path/images, input_folder_path/detections, and input_folder_path/statistics
    """
    statistics_folder = os.path.join(output_folder, 'statistics')
    detections_folder = os.path.join(output_folder, 'detections')
    images_folder = os.path.join(output_folder, 'images')
    return images_folder, detections_folder, statistics_folder


def check_path_is_directory(path: str) -> str | None:
    """
    Returns None if the path does not exist or is not a directory; otherwise, the path is returned
    :param path: The path to check
    """
    if not os.path.isdir(path):
        return None
    return path


def make_parser() -> argparse.ArgumentParser:
    """
    Creates the parser used for this application, and configures the required images.
    :return: The argument parser
    """
    parser = argparse.ArgumentParser(description='Tycho CDM')
    parser.add_argument('-i', '--input', dest='input_folder', type=str, required=True,
                        help='Path to input folder')
    parser.add_argument('-o', '--output', dest='output_folder', type=str, required=True,
                        help='Path to output folder')
    parser.add_argument('-p', '--planet_name', dest='planet_name', type=str, required=True,
                        help='Name of the planet, either \'mars\' or \'moon\'')
    return parser


def dir_is_empty(path: str) -> bool:
    """
    Checks if the given directory is empty
    If the given path does not point at a directory, a NotADirectory exception is thrown instead.
    :param path: THe path to check
    :return: True if the given directory is empty; False otherwise.
    """
    return not os.listdir(path)


def write_results(results, labels_directory, metadata_directory, output_folder_path) -> None:
    """
    After inference, this method handles generation and output of the required results
    :param results: The results from inference, in x1,y1,x2,y2 (i.e. top left, bottom right) format
    :param labels_directory: Optional, location of ground truth data
    :param metadata_directory: Optional, location of additional information to compute crater diameters and positions
    :param output_folder_path: Location to store the outputs in
    """

    # Create mandatory output subdirectories
    images_folder, detections_folder, statistics_folder = get_output_directories(output_folder_path)

    os.makedirs(detections_folder)
    os.makedirs(images_folder)
    os.makedirs(statistics_folder)

    # Get paths to each label and metadata file, if present
    label_file_paths = sorted(glob.glob(os.path.join(labels_directory, '*'))) \
        if labels_directory is not None else None
    metadata_file_paths = sorted(glob.glob(os.path.join(metadata_directory, '*'))) \
        if metadata_directory is not None else None

    all_predicted_bboxes = []
    for i, (image_path, bboxes, _, confidences) in enumerate(results):
        file_name = Path(image_path).name[:-4]
        image = cv2.imread(image_path)

        # Create a visualization of bounding boxes over each image (and ground truth boxes, if applicable)
        visualizer.visualize(
            image_path,
            bboxes,
            confidences,
            images_folder,
            label_file_paths[i] if label_file_paths is not None else None)

        # Convert x1,y1,x2,y2 output of object detection model into fractional x,y,w,h
        bboxes_xywh = xyxy2xywh(bboxes).astype(np.float64)
        bboxes_xywh[:, 0] /= float(image.shape[1])
        bboxes_xywh[:, 2] /= float(image.shape[1])
        bboxes_xywh[:, 1] /= float(image.shape[0])
        bboxes_xywh[:, 3] /= float(image.shape[0])

        all_predicted_bboxes.append(bboxes_xywh)

        # Write the bounding boxes for this image to a .csv file in detections/
        # If metadata was given, this also writes crater position and diameter
        with open(os.path.join(detections_folder, f'{file_name}.csv'), 'w') as bbox_file:
            if metadata_file_paths is not None:
                metadata = pd.read_csv(metadata_file_paths[i], header=None).to_numpy()
                if metadata.shape != (1, 5):
                    raise RuntimeError(
                        f"Expected {metadata_file_paths[i]} to have shape (1, 5), but was {metadata.shape}; see README")
                metadata = metadata[0]
                lats, longs, diameters = get_lat_long_and_diameter(bboxes_xywh, image.shape[0], image.shape[1],
                                                                   metadata[0], metadata[1], metadata[2], metadata[3],
                                                                   metadata[4])

                metadata_df = pd.DataFrame([lats, longs, diameters]).T
                bboxes_df = pd.DataFrame(bboxes_xywh)
                pd.concat([bboxes_df, metadata_df], axis=1).to_csv(bbox_file, header=False, index=False)
            else:
                pd.DataFrame(bboxes_xywh).to_csv(bbox_file, header=False, index=False)

        if metadata_file_paths is not None:
            plot_distribution_graph(statistics_folder, file_name, bboxes_xywh, diameters, metadata[4])
        else:
            plot_distribution_graph(statistics_folder, file_name, bboxes_xywh)

    # If we were given labels, then statistics can be calculated here
    if labels_directory is not None and labels_directory != "":
        statistics_file = os.path.join(statistics_folder, 'statistics.csv')
        all_true_bboxes = []
        for label_file in label_file_paths:
            all_true_bboxes.append(np.loadtxt(label_file, delimiter=','))
        try:
            TP, FP, FN = metric.read_boxes(all_predicted_bboxes, all_true_bboxes)
            pd.DataFrame([TP, FP, FN], index=['TP', 'FP', 'FN']).T.to_csv(statistics_file, index=True)
        except Exception as e:
            pass


if __name__ == '__main__':
    main()  # Don't run this directly - use the command line or GUI instead. See README file.
