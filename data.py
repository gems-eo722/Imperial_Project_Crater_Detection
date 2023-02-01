"""Input Image Dataset Generator

Script for generating input and target image datasets.
"""

import pandas as pd
from sklearn.cluster import KMeans
import cv2
import os
import numpy as np
from math import cos,radians

def iou(BBGT, imgRect):
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom - left_top, 0)
    inter_area = wh[:, 0] * wh[:, 1]
    iou = inter_area / ((BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) + 0.0000001)
    return iou


def project(phi, lamda, lamda0):
    y = (lamda - lamda0) * cos(radians(phi))
    return y

class MoonDataset:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df_s = df[["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"]].dropna()
        self.data = np.asarray(df_s)
        self.labels = None
        self.extents = [[-90, -180, 0, -45], [-90, -180, 45, 0], [0, -90, 0, -45], [0, -90, 45, 0]]
        self.Names = ["A", "B", "C", "D"]

        self.shapes = []
        for name in self.Names:
            img = cv2.imread(os.path.join("Moon_WAC_Training/images", "Lunar_" + name + ".jpg"))
            self.shapes.append(img.shape)

        self.ratios_w = []
        self.ratios_h = []
        for i, extent in enumerate(self.extents):
            h, w = self.shapes[i][:-1]
            w_ratio = abs(extent[0] - extent[1]) / w
            h_ratio = abs(extent[2] - extent[3]) / h
            self.ratios_w.append(w_ratio)
            self.ratios_h.append(h_ratio)
        self.genClass(3)

    def genClass(self, n_cluster):
        l = self.data[:, -1].reshape(-1, 1)
        self.labels = KMeans(n_clusters=n_cluster).fit_predict(l)

    # Generate coordinates based on the label dataset using latitude/longitude projection
    def genCoordFile(self):
        file_os = []
        for i in range(4):
            file_os.append(open(self.Names[i] + ".csv", "w"))

        for i, (lat, long, l) in enumerate(self.data):
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

            extent = self.extents[region]
            long0 = 0
            long = project(lat, long, long0)
            h, w = self.shapes[region][:-1]
            w_ratio = self.ratios_w[region]
            h_ratio = self.ratios_h[region]
            x = int(abs(long - extent[1]) / w_ratio)
            y = int(abs(lat - extent[2]) / h_ratio)
            radius = int(5 * l)
            x1, y1, x2, y2 = x - radius, y - radius, x + radius, y + radius
            file_os[region].write("%d,%d,%d,%d,%d\n" % (
                x1 if x1 > 0 else 0, y1 if y1 > 0 else 0, x2 if x2 < h else h - 1, y2 if y2 < w else w - 1,
                self.labels[i]))
        for i in range(4):
            file_os[i].close()

    # Split the input image into smaller tiles
    def split(self, lable, subsize=416, iou_thresh=0.2, gap=50):
        if not isinstance(lable, list):
            lable = [lable]
        for name in self.Names:
            dirdst = os.path.join("data", name)
            BBGT = np.asarray(pd.read_csv(os.path.join("data", name + ".csv")))
            img = cv2.imread(os.path.join("Moon_WAC_Training/images", "Lunar_" + name + ".jpg"))
            img_h, img_w = img.shape[:2]
            top = 0
            reachbottom = False
            while not reachbottom:
                reachright = False
                left = 0
                if top + subsize >= img_h:
                    reachbottom = True
                    top = max(img_h - subsize, 0)
                while not reachright:
                    if left + subsize >= img_w:
                        reachright = True
                        left = max(img_w - subsize, 0)
                    imgsplit = img[top:min(top + subsize, img_h), left:min(left + subsize, img_w)]
                    if imgsplit.shape[:2] != (subsize, subsize):
                        template = np.zeros((subsize, subsize, 3), dtype=np.uint8)
                        template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                        imgsplit = template
                    imgrect = np.array([left, top, left + subsize, top + subsize]).astype('float32')
                    ious = iou(BBGT[:, :4].astype('float32'), imgrect)
                    BBpatch = BBGT[ious > iou_thresh]
                    BBpatch_LABELD = BBpatch[np.in1d(BBpatch[:, -1], lable)]
                    ## abandaon images with 0 bboxes
                    if len(BBpatch_LABELD) > 0:
                        split_name = os.path.join(dirdst, name + '_' + str(left) + '_' + str(top))
                        cv2.imwrite(split_name + ".png", imgsplit)
                        f = open(split_name + '.txt', "w")
                        for bb in BBpatch_LABELD:
                            x1, y1, x2, y2, target_id = int(bb[0]) - left, int(bb[1]) - top, int(bb[2]) - left, int(
                                bb[3]) - top, int(bb[4])
                            f.write("%d %d %d %d %d\n" % (0,
                                                          x1 if x1 > 0 else 0, y1 if y1 > 0 else 0,
                                                          x2 if x2 < subsize else subsize - 1,
                                                          y2 if y2 < subsize else subsize - 1))
                        f.close()
                    left += subsize - gap
                top += subsize - gap

s = MoonDataset("Moon_WAC_Training/labels/lunar_crater_database_robbins_train.csv")
s.genCoordFile()
s.split([0,1,2])

for root, dirs, files in os.walk("data/A"):
    for file in files:
        if file.endswith("png"):
            name = file.split(".")[0]
            img_path = os.path.join(root, file)
            visual_path = os.path.join("data/visual", file)
            label_path = os.path.join(root, name + ".txt")
            img = cv2.imread(img_path)
            f = open(label_path, "r")
            for l in f:
                try:
                    _,x1, y1, x2, y2 = l.strip().split(" ")
                except:
                    print(img_path)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.imwrite(visual_path,img)
