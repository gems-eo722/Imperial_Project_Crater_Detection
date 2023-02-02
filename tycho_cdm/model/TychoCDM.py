import os
from pathlib import Path

import mmcv
import numpy
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

from mmyolo.utils import register_all_modules
from tycho_cdm.visualization.worker import Worker


class TychoCDM:

    def __init__(self, planet_name):
        config_path = Path(os.path.realpath(__file__)).parent.joinpath('configs/yolov5_m_v61.py')

        mars_path = Path(os.path.realpath(__file__)).parent.joinpath('weights/epoch_80.pth').__str__()  # TODO
        moon_path = Path(os.path.realpath(__file__)).parent.joinpath('weights/epoch_80.pth').__str__()  # TODO
        self.weights_file_path = \
            mars_path if planet_name.lower() == 'mars' \
                else (moon_path if planet_name.lower() == 'moon' else None)

        if self.weights_file_path is None:
            raise RuntimeError(f"Invalid planet name: {planet_name}")

        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        register_all_modules()
        self.model = init_detector(self.config_path, self.weights_file_path, device=self.device)
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

    def batch_inference(self, batch_img_path, gui_worker: Worker = None):
        img_name_list = os.listdir(batch_img_path)

        results = []
        for i, img_name in enumerate(img_name_list):
            img_path = os.path.join(batch_img_path, img_name)
            bboxes, labels, scores = self.single_inference(img_path)
            results.append((img_path, bboxes, labels, scores))
            if gui_worker is not None:
                gui_worker.progress.emit(i + 1)
                if gui_worker.shouldClose:
                    gui_worker.finished.emit()
                    return []

        if gui_worker is not None:
            gui_worker.finished.emit()
        return results

    def inference(self, image):
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
        return (bbox, label, score)
