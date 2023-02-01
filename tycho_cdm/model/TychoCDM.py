import os
from pathlib import Path

import mmcv
import numpy
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

from mmyolo.utils import register_all_modules


class TychoCDM:

    def __init__(self, weights_file_path):
        config_path = Path(os.path.realpath(__file__)).parent.joinpath('configs/yolov5_m_v61.py')
        self.weights_file_path = weights_file_path
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TODO - load object detection model with weights here
        register_all_modules()
        self.model = init_detector(self.config_path, weights_file_path, device=self.device)
        print('initializing model')

    def single_inference(self, img_path):
        image = mmcv.imread(img_path, channel_order='rgb')
        result = inference_detector(self.model, image)
        score_result = result.pred_instances['scores']
        bbox_result = result.pred_instances['bboxes']
        label_result = result.pred_instances['labels']
        index = score_result > 0.3
        if self.device == 'cpu':
            bbox = bbox_result[index].detach().numpy()
            label = label_result[index].detach().numpy()
            score = score_result[index].detach().numpy()
        else:
            bbox = bbox_result[index].detach().cpu().numpy()
            label = label_result[index].detach().cpu().numpy()
            score = score_result[index].detach().cpu().numpy()
        new_bbox = numpy.zeros_like(bbox)
        new_bbox[:, 2] = np.absolute(bbox[:, 0] - bbox[:, 2]) / image.shape[1]
        new_bbox[:, 3] = np.absolute(bbox[:, 1] - bbox[:, 3]) / image.shape[0]
        new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / (2 * image.shape[1])
        new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / (2 * image.shape[0])
        return new_bbox, label, score

    def batch_inference(self, batch_img_path):
        img_name_list = os.listdir(batch_img_path)
        bbox_result_list = []
        label_result_list = []
        confident_result_list = []
        for img_name in img_name_list:
            img_path = os.path.join(batch_img_path, img_name)
            bbox, label, score = self.single_inference(img_path)
            bbox_result_list.append(bbox)
            label_result_list.append(label)
            confident_result_list.append(score)
        return bbox_result_list, label_result_list, confident_result_list

    def predict(self, image_path, label_path=None, data_path=None) -> any:  # TODO - return type

        # TODO - calculate bounding boxes here
        #   * if label_path is given, also return statistics (FNs, TPs, FPs)
        #   * if data_path is given, output for each image a .csv file with (lat,long) position
        #   and diameter (in km) of each crater
        # returns (bounding_boxes, statistics, crater_data)
        # `statistics` and `crater_data` may be `None` if inputs didn't provide necessary information
        pass
