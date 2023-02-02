import cv2


def convert2loc(imgpath, long, lat, hdegree, wdegree, box, resolution):
    x_c = (box[0] + box[2]) / 2
    y_c = (box[1] + box[3]) / 2
    img = cv2.imread(imgpath)
    h, w = img.shape[:-1]
    lat_start = lat - 0.5 * wdegree
    long_start = long - 0.5 * hdegree
    h_ratio = hdegree / h
    w_ratio = wdegree / w
    c_long = x_c * w_ratio + long_start
    c_lat = y_c * h_ratio + lat_start
    diameter = (abs(box[0] - box[2]) + abs(box[1] + box[3])) / 2 * resolution / 1000
    return c_lat, c_long, diameter

