import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tycho_cdm.model.TychoCDM import TychoCDM
from tycho_cdm.visualization import visualizer


def main():
    parser = make_parser()
    images_folder, labels_folder, metadata_folder, planet_name, output_folder = parse_arguments(parser)

    model = TychoCDM(planet_name)
    results = model.batch_inference(images_folder)

    write_results(results, labels_folder, metadata_folder, output_folder)


def run_batch(images_directory, labels_directory, metadata_directory, planet_name) -> any:  # TODO - return type
    image_file_paths = sorted(glob.glob(os.path.join(images_directory, '*')))
    label_file_paths = sorted(glob.glob(os.path.join(labels_directory, '*'))) \
        if labels_directory is not None else []
    metadata_file_paths = sorted(glob.glob(os.path.join(metadata_directory, '*'))) \
        if metadata_directory is not None else []

    has_labels = len(label_file_paths) > 0
    has_metadata = len(metadata_file_paths) > 0

    if has_labels and len(label_file_paths) != len(image_file_paths):
        raise RuntimeError("Number of label files does not match number of images")
    if has_metadata and len(metadata_file_paths) != len(image_file_paths):
        raise RuntimeError("Number of metadata files does not match number of images")

    model = TychoCDM(planet_name)
    model.batch_inference(images_directory)


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    input_path: str = args.input_path
    output_path: str = args.output_path
    planet_name: str = args.planet_name

    return process_arguments(input_path, output_path, planet_name)


def process_arguments(input_path: str, output_path: str, planet_name: str):
    labels_path = os.path.join(input_path, 'labels')
    metadata_path = os.path.join(input_path, 'data')
    images_path = os.path.join(input_path, 'images')

    if not os.path.isdir(input_path):
        raise RuntimeError(f'Given path to input directory is invalid: {input_path}')

    if not os.path.isdir(labels_path):
        labels_path = None

    if not os.path.isdir(metadata_path):
        metadata_path = None

    if not os.path.isdir(images_path):
        raise RuntimeError('Input directory does not contain \'images\' subdirectory')
    if dir_is_empty(images_path):
        raise RuntimeError('\'image\' subdirectory is empty, nothing to do')

    if (os.path.isdir(output_path)) and not dir_is_empty(output_path):
        raise RuntimeError('Output directory exists and is not empty')

    if planet_name.lower() != 'moon' and planet_name.lower() != 'mars':
        raise RuntimeError('The planet name must be \'mars\' or \'moon\'')

    return images_path, labels_path, metadata_path, planet_name, output_path


def make_parser():
    parser = argparse.ArgumentParser(description='Tycho CDM')
    parser.add_argument('-i', '--input', dest='input_path', type=str, required=True,
                        help='Path to input folder')
    parser.add_argument('-o', '--output', dest='output_path', type=str, required=True,
                        help='Path to output folder')
    parser.add_argument('-p', '--planet_name', dest='planet_name', type=str, required=True,
                        help='Name of the planet, either \'mars\' or \'moon\'')
    return parser


def dir_is_empty(path: str) -> bool:
    return not os.listdir(path)


def plot_distribution_graph(folder_path, file_name, bboxes, metadata=None):
    """
    >>> boxes = np.random.random((10000, 4))
    >>> plot_distribution_graph('./', 'f', boxes)
    """
    widths = [box[2] for box in bboxes]
    heights = [box[3] for box in bboxes]

    metres_per_pixel = None
    if metadata is None:
        metres_per_pixel = 100
    else:
        pass  # TODO - set metres per pixel from metadata

    sns.set()
    if metadata is None:
        # No metadata, calculate diameter as mean of width and height of bounding rect
        plot = sns.displot(
            ([(w + h) / 2000 for (w, h) in (zip(widths * metres_per_pixel, heights * metres_per_pixel))]),
            kind='hist', log_scale=10, aspect=2)

        plt.xscale('log')
        plt.yscale('log')
    else:
        plot = None  # TODO - plot actual diameters, contained in metadata

    plt.xlabel("Crater Diameter (km)")
    plt.title("Size-Frequency Distribution of Crater Sizes")

    plot.savefig(os.path.join(folder_path, f'{file_name}.png'))
    plt.close(plot.fig)


def write_results(results, labels_directory, metadata_directory, output_folder_path):
    # Create mandatory output subdirectories
    detections_path = os.path.join(output_folder_path, 'detections')
    output_images_path = os.path.join(output_folder_path, 'images')
    statistics_path = os.path.join(output_folder_path, 'statistics')
    os.makedirs(detections_path)
    os.makedirs(output_images_path)
    os.makedirs(statistics_path)

    # Get paths to each label and metadata file, if present
    label_file_paths = sorted(glob.glob(os.path.join(labels_directory, '*'))) \
        if labels_directory is not None else None
    metadata_file_paths = sorted(glob.glob(os.path.join(metadata_directory, '*'))) \
        if metadata_directory is not None else None

    for i, (image_path, bboxes, _, confidences) in enumerate(results):
        file_name = Path(image_path).name[:-4]

        # Create a visualization of bounding boxes over each image (and ground truth boxes, if applicable)
        visualizer.visualize(
            image_path,
            bboxes,
            confidences,
            output_images_path,
            label_file_paths[i] if label_file_paths is not None else None)

        # Write the bounding boxes for this image to a .csv file in detections/
        # If metadata was given, this also writes crater position and diameter
        with open(os.path.join(detections_path, f'{file_name}.csv'), 'w') as bbox_file:
            if metadata_file_paths is not None:
                # (lat, long), size = something(bboxes, ref_data_paths[i])
                pass  # TODO - append lat,long,diameter to bboxes array
            pd.DataFrame(bboxes).to_csv(bbox_file, header=False, index=False)

        if metadata_file_paths is not None:
            plot_distribution_graph(statistics_path, file_name, bboxes, metadata_file_paths[i])
        else:
            plot_distribution_graph(statistics_path, file_name, bboxes)

    # If we were given labels, then statistics can be calculated here
    if labels_directory is not None and labels_directory != "":
        pass  # TODO - output stats (TP, FN, FP) - only 1 stats file for ALL images


if __name__ == '__main__':
    main()  # Don't run this directly - use the command line or GUI instead. See README file.
