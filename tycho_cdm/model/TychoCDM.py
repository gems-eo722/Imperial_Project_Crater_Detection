class TychoCDM:

    def __init__(self, weights_file_path):
        self.weights_file_path = weights_file_path
        # TODO - load object detection model with weights here

    def predict(self, image_path, label_path=None, data_path=None) -> any:  # TODO - return type
        # TODO - calculate bounding boxes here
        #   * if label_path is given, also return statistics (FNs, TPs, FPs)
        #   * if data_path is given, output for each image a .csv file with (lat,long) position
        #   and diameter (in km) of each crater
        # returns (bounding_boxes, statistics, crater_data)
        # `statistics` and `crater_data` may be `None` if inputs didn't provide necessary information
        pass
