import argparse
import os

from tycho_cdm.model.TychoCDM import TychoCDM


def main():
    parser = make_parser()
    weights_file_path, images_path, labels_path, additional_data_path = parse_arguments(parser)

    images = os.listdir(images_path)
    labels = os.listdir(labels_path) if labels_path is not None else []
    data = os.listdir(additional_data_path) if additional_data_path is not None else []

    has_labels = len(labels) > 0
    has_data = len(data) > 0

    if has_labels and len(labels) != len(images):
        parser.error("Number of label files does not match number of images")
    if has_data and len(data) != len(images):
        parser.error("Number of data files does not match number of images")

    model = TychoCDM(weights_file_path)
    results = []
    for i in range(len(images)):
        results.append(model.predict(images[i], labels[i] if has_labels else None, data[i] if has_data else None))

    for result in results:
        bounding_boxes, statistics, crater_data = result
        pass  # TODO - generate bounding box visualizations, make plots, etc. (output / visualization)


def parse_arguments(parser: argparse.ArgumentParser) -> tuple[str, str, str, str]:
    args = parser.parse_args()

    weights_file_path: str = args.weights_file_path
    if not os.path.isfile(weights_file_path):
        parser.error(f'Given path to weights file is invalid: {weights_file_path}')

    if not os.path.isdir(args.input_path):
        parser.error(f'Given path to input directory is invalid: {args.input_path}')

    labels_path = os.path.join(args.input_path, 'labels')
    if not os.path.isdir(labels_path):
        labels_path = None

    data_path = os.path.join(args.input_path, 'data')
    if not os.path.isdir(data_path):
        data_path = None

    images_path = os.path.join(args.input_path, 'images')
    if not os.path.isdir(images_path):
        parser.error('Input directory does not contain \'images\' subdirectory')
    if dir_is_empty(images_path):
        parser.error('\'image\' subdirectory is empty, nothing to do')

    if not os.path.isdir(args.output_path):
        parser.error(f'Given path to output directory is invalid: {args.output_path}')
    if not dir_is_empty(args.output_path):
        parser.error('Output directory is not empty')

    if args.planet_name.lower() != 'moon' and args.planet_name.lower() != 'mars':
        parser.error('The planet name must be \'mars\' or \'moon\'')

    return weights_file_path, images_path, labels_path, data_path


def make_parser():
    parser = argparse.ArgumentParser(description='Tycho CDM')
    parser.add_argument('-w', '--weights', dest='weights_file_path', type=str, required=True,
                        help='Path to weights file for object detection model')
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
