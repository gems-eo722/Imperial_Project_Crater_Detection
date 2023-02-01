import argparse
import os

from tycho_cdm.model.TychoCDM import TychoCDM
from tycho_cdm.visualization import visualizer


def main():
    parser = make_parser()
    images_path, labels_path, data_path, planet_name, output_path = parse_arguments(parser)

    model = TychoCDM(planet_name)
    results = model.batch_inference(images_path)

    write_results(results, labels_path, data_path, output_path)


def run_batch(images_path, labels_path, data_path, planet_name) -> any:  # TODO - return type
    images = os.listdir(images_path)
    labels = os.listdir(labels_path) if labels_path is not None else []
    data = os.listdir(data_path) if data_path is not None else []

    has_labels = len(labels) > 0
    has_data = len(data) > 0

    if has_labels and len(labels) != len(images):
        raise RuntimeError("Number of label files does not match number of images")
    if has_data and len(data) != len(images):
        raise RuntimeError("Number of data files does not match number of images")

    model = TychoCDM(planet_name)
    model.batch_inference(images_path)


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    input_path: str = args.input_path
    output_path: str = args.output_path
    planet_name: str = args.planet_name

    return process_arguments(input_path, output_path, planet_name)


def process_arguments(input_path: str, output_path: str, planet_name: str):
    labels_path = os.path.join(input_path, 'labels')
    data_path = os.path.join(input_path, 'data')
    images_path = os.path.join(input_path, 'images')

    if not os.path.isdir(input_path):
        raise RuntimeError(f'Given path to input directory is invalid: {input_path}')

    if not os.path.isdir(labels_path):
        labels_path = None

    if not os.path.isdir(data_path):
        data_path = None

    if not os.path.isdir(images_path):
        raise RuntimeError('Input directory does not contain \'images\' subdirectory')
    if dir_is_empty(images_path):
        raise RuntimeError('\'image\' subdirectory is empty, nothing to do')

    if (os.path.isdir(output_path)) and not dir_is_empty(output_path):
        raise RuntimeError('Output directory exists and is not empty')

    if planet_name.lower() != 'moon' and planet_name.lower() != 'mars':
        raise RuntimeError('The planet name must be \'mars\' or \'moon\'')

    return images_path, labels_path, data_path, planet_name, output_path


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


if __name__ == '__main__':
    main()


def write_results(results, labels_path, data_path, output_folder_path):
    detections_path = os.path.join(output_folder_path, 'detections')
    output_images_path = os.path.join(output_folder_path, 'images')
    os.makedirs(detections_path)
    os.makedirs(output_images_path)

    labels = os.listdir(labels_path) if labels_path is not None else None
    for i, (image_path, bboxes, _, confidence) in enumerate(results):
        visualizer.visualize(
            image_path,
            bboxes,
            output_images_path,
            os.path.join(labels_path, labels[i]) if labels is not None else None)

    if labels_path is not None and labels_path != "":
        statistics_path = os.path.join(output_folder_path, 'statistics')
        os.makedirs(statistics_path)
        # todo output stats (Precision, Recall, F1, IoU)
        # todo output distribution graph

    if data_path is not None and data_path != "":
        crater_data_path = os.path.join(output_folder_path, 'crater_data')
        os.makedirs(crater_data_path)
        pass  # todo output 1 .csv file per image, which has position and size of each crater in the image
