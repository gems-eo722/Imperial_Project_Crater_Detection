import cv2
import os
import pandas as pd
import numpy as np

img_size = 416

# for name in os.listdir("Moon_WAC_Training/images"):
#     img_file = os.path.join("Moon_WAC_Training/images", name)
#     label_file = os.path.join("Moon_WAC_Training/labels", os.path.basename(name).split(".png")[0] + ".csv")
#     f = open(label_file, "r")
#     img = cv2.imread(img_file)
#     for lines in f:
#         bbox = lines.strip().split(",")
#         bbox = [int(float(x) * img_size) for x in bbox]
#         x, y, w, h = bbox
#         cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2, 4)
#     cv2.imwrite(os.path.join("moon",name),img)

df = pd.read_csv("Moon_WAC_Training/labels/lunar_crater_database_robbins_train.csv")
df_s = df[["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"]].dropna()
data = np.asarray(df_s)

extents = [[-90, -180, 0, -45], [-90, -180, 45, 0], [0, -90, 0, -45], [0, -90, 45, 0]]
Names = ["A", "B", "C", "D"]
shapes = []
imgs = []
for name in Names:
    img = cv2.imread(os.path.join("Moon_WAC_Training/images","Lunar_" + name + ".jpg"))
    shapes.append(img.shape)
    imgs.append(img)

ratios_w = []
ratios_h = []
for i, extent in enumerate(extents):
    w, h = shapes[i][:-1]
    w_ratio = abs(extent[0] - extent[1]) / w
    h_ratio = abs(extent[2] - extent[3]) / h
    ratios_w.append(w_ratio)
    ratios_h.append(h_ratio)

for lat, long, l in data:
    if long > -90:
        if lat > 0:
            region = 3
        else:
            region = 2
    else:
        if lat > 0:
            region = 1
        else:
            region = 0
    img = imgs[region]
    extent = extents[region]
    w_ratio = ratios_w[region]
    h_ratio = ratios_h[region]
    x = int(abs(long - extent[1]) / w_ratio)
    y = int(abs(lat - extent[3]) / h_ratio)
    radius = int(5 * l)
    cv2.rectangle(img, (int(x - radius), int(y - radius)), (int(x + radius), int(y + radius)), (0, 255, 255), 2, 4)

for i, img in enumerate(imgs):
    name = Names[i]
    cv2.imwrite(os.path.join("moon", "Lunar_"+name+'.png'), img)
