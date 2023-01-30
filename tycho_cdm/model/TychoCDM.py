class TychoCDM:

    def __init__(self, weights_file_path):
        self.weights_file_path = weights_file_path
        # TODO - load object detection model with weights here

    def predict(self, image_path, label_path=None, data_path=None):
        # TODO - calculate bounding boxes here
        #   * if label_path is given, also return statistics (FNs, TPs, FPs)
        #   * if data_path is given, output for each image a .csv file with (lat,long) position
        #   and diameter (in km) of each crater
        pass
