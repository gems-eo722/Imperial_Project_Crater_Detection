import argparse
import os

from tycho_cdm.model.TychoCDM import TychoCDM


def main():
    parser = make_parser()
    weights_file_path, images_path, labels_path, data_path, planet_name, output_path = parse_arguments(parser)

    model = TychoCDM(weights_file_path)
    results = model.batch_inference(images_path)
    # results = run_batch(weights_file_path, images_path, labels_path, data_path, planet_name)

    for result in results:
        # bounding_boxes, statistics, crater_data = result
        pass  # TODO - generate bounding box visualizations, make plots, etc. (output / visualization)


def run_batch(weights_file_path, images_path, labels_path, data_path, planet_name) -> any:  # TODO - return type
    images = os.listdir(images_path)
    labels = os.listdir(labels_path) if labels_path is not None else []
    data = os.listdir(data_path) if data_path is not None else []

    has_labels = len(labels) > 0
    has_data = len(data) > 0

    if has_labels and len(labels) != len(images):
        raise RuntimeError("Number of label files does not match number of images")
    if has_data and len(data) != len(images):
        raise RuntimeError("Number of data files does not match number of images")

    model = TychoCDM(weights_file_path)
    model.batch_inference(images_path)
    # results = []
    # for i in range(len(images)):
    #     results.append(
    #         model.predict(
    #             os.path.join(images_path, images[i]),
    #             os.path.join(labels_path, labels[i]) if has_labels else None,
    #             os.path.join(data_path, data[i]) if has_data else None))
    #
    # return results


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    input_path: str = args.input_path
    output_path: str = args.output_path
    weights_file_path: str = args.weights_file_path
    planet_name: str = args.planet_name

    return process_arguments(weights_file_path, input_path, output_path, planet_name)


def process_arguments(weights_file_path: str, input_path: str, output_path: str, planet_name: str):
    labels_path = os.path.join(input_path, 'labels')
    data_path = os.path.join(input_path, 'data')
    images_path = os.path.join(input_path, 'images')

    if not os.path.isfile(weights_file_path):
        raise RuntimeError(f'Given path to weights file is invalid: {weights_file_path}')

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

    return weights_file_path, images_path, labels_path, data_path, planet_name, output_path


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


def write_results(results, labels_path, data_path, planet_name, output_folder_path):
    bboxes, _, confidences = results
