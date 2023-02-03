def get_lat_long_and_diameter(boxes, image_height, image_width, long, lat, horizontal_degree, vertical_degree, resolution=100) -> tuple[list, list, list]:
    """
    Calculates diameters and (lat,long) positions for bounding boxes in an image
    :param boxes: The bounding boxes of the image
    :param image_height:
    :param image_width:
    :param long: The image center longitude
    :param lat: The image center latitude
    :param horizontal_degree: The image width in degrees
    :param vertical_degree: The image height in degrees
    :param resolution: The image resolution in metres per pixel
    :return A tuple of three lists, for latitudes, longitudes, and diameters of the bounding boxes
    """
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
