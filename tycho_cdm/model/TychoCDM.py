import glob
import os
from pathlib import Path

import mmcv
import numpy
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

from mmyolo.utils import register_all_modules
from tycho_cdm.utils.post_process import inference
from tycho_cdm.visualization.worker import Worker


class TychoCDM:
    """
    Object detection model of Tycho; based on YOLO v5/v8
    """
    def __init__(self, planet_name):
        """
        Constructs an instance of the object detection model.
        :param planet_name: The planet to predict craters on; 
        this parameter dictates the config and weight file to be loaded
        """
        mars: bool = planet_name.lower() == 'mars'
        moon: bool = planet_name.lower() == 'moon'
        if not mars and not moon:
            raise RuntimeError(f"Given planet name is invalid: {planet_name}")

        # Get config for planet
        mars_config_path = Path(os.path.realpath(__file__)).parent.joinpath('configs/mars_config.py')
        moon_config_path = Path(os.path.realpath(__file__)).parent.joinpath('configs/moon_config.py')
        self.config_path = mars_config_path if mars else moon_config_path

        if not os.path.isfile(self.config_path):
            raise RuntimeError(f"Config file for {planet_name} not found, please check README.")

        # Get weights for planet
        mars_path = Path(os.path.realpath(__file__)).parent.joinpath('weights/mars_weights.pth').__str__()
        moon_path = Path(os.path.realpath(__file__)).parent.joinpath('weights/moon_weights.pth').__str__()
        self.weights_file_path = mars_path if mars else moon_path

        # Check that user installed weights
        if not os.path.isfile(self.weights_file_path):
            raise RuntimeError(f"Weights file for {planet_name} not found, please check README.")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        register_all_modules()
        self.model = init_detector(self.config_path, self.weights_file_path, device=self.device)
        print('Initializing model')

    def single_inference(self, image) -> tuple[list, list, list]:
        """
        Performs crater detection for a single image.
        :param image: The image to detect craters in
        :return: The bounding boxes, labels (always "crater"), and confidences for each bounding box
        """
        result = inference_detector(self.model, image)
        score_result = result.pred_instances['scores']
        bbox_result = result.pred_instances['bboxes']
        label_result = result.pred_instances['labels']
        index = score_result > 0.3
        bbox, label, score = self.results_to_numpy(bbox_result, index, label_result, score_result)
        new_bbox = numpy.zeros_like(bbox)
        new_bbox[:, 2] = np.absolute(bbox[:, 0] - bbox[:, 2]) / image.shape[1]
        new_bbox[:, 3] = np.absolute(bbox[:, 1] - bbox[:, 3]) / image.shape[0]
        new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / (2 * image.shape[1])
        new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / (2 * image.shape[0])
        return new_bbox, label, score

    def batch_inference(self, batch_img_path, gui_worker: Worker = None) -> list[tuple[str, list, list, list]]:
        """
        Performs crater detection for a batch of images. The inference method for a single image
        is sensitive to image width, height, and resolution, and may split the input image if it exceeds
        the maximum size for a single image.
        :param batch_img_path: The path to the folder containing all images
        :param gui_worker: Optional, a PyQt GUI thread to update the GUI progress bar
        :return A list containing for each image the bounding boxes, labels (always "crater"),
        and confidences for each bounding box in that image
        """
        img_name_list = sorted(glob.glob(os.path.join(batch_img_path, '*')))

        results = []
        for i, img_path in enumerate(img_name_list):
            image = mmcv.imread(img_path, channel_order='rgb')

            # Perform inference on image (if image is large, may split image, infer each, then collect and return)
            bboxes, labels, scores = inference(image, self)
            results.append((img_path, bboxes, labels, scores))

            # Update progress bar in GUI
            if gui_worker is not None:
                gui_worker.progress.emit(i + 1)
                if gui_worker.shouldClose:
                    gui_worker.finished.emit()
                    return []

        if gui_worker is not None:
            gui_worker.finished.emit()
        return results

    def split_batch_inference_(self, images, iou_threshold):
        """
        Performs batch inference for a set of images obtained by splitting a larger image.
        :param images: The images to detect craters in
        :param iou_threshold: Determines if a box is included in the result, based on the IoU metric
        :return: For the big image from which the split was derived, this method returns the bounding boxes,
        labels (always "crater"), and confidences for each bounding box
        """
        results = inference_detector(self.model, images)
        bbox_list = []
        score_list = []
        label_list = []
        for result in results:
            score_result = result.pred_instances['scores']
            bbox_result = result.pred_instances['bboxes']
            label_result = result.pred_instances['labels']
            index = score_result > iou_threshold
            bbox, label, score = self.results_to_numpy(bbox_result, index, label_result, score_result)
            bbox_list.append(bbox)
            score_list.append(score)
            label_list.append(label)
        return bbox_list, label_list, score_list

    def results_to_numpy(self, bbox_result, index, label_result, score_result):
        """
        Converts a single bounding box from the CDM to numpy, and detaches the tensor if running on the cpu
        :param bbox_result: The resulting bounding box
        :param index: The positions to select for the result, based on IoU
        :param label_result: Labels of the result
        :param score_result: Confidence scores of the result
        :return The bounding box, label (always "crater"), and confidence for this bounding box
        """
        if self.device == 'cpu':
            bbox = bbox_result[index].detach().numpy()
            label = label_result[index].detach().numpy()
            score = score_result[index].detach().numpy()
        else:
            bbox = bbox_result[index].detach().cpu().numpy()
            label = label_result[index].detach().cpu().numpy()
            score = score_result[index].detach().cpu().numpy()
        return bbox, label, score
