import cv2


def convert2loc(boxes, image_height, image_width, long, lat, horizontal_degree, vertical_degree, resolution=100):
    c_lats = []
    c_longs = []
    diameters = []
    for box in boxes:
        x_c = (box[0] + box[2]) / 2
        y_c = (box[1] + box[3]) / 2
        lat_start = lat - 0.5 * vertical_degree
        long_start = long - 0.5 * horizontal_degree
        h_ratio = horizontal_degree / image_height
        w_ratio = vertical_degree / image_width
        c_long = x_c * w_ratio + long_start
        c_lat = y_c * h_ratio + lat_start
        diameter = (abs(box[0] - box[2]) + abs(box[1] + box[3])) / 2 * resolution / 1000
        c_longs.append(c_long)
        c_lats.append(c_lat)
        diameters.append(diameter)
    return c_lats, c_longs, diameters
