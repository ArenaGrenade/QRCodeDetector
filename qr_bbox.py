from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import cdist  
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import random
import math
import time
import cv2
import os
import re

def find_slope_and_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        return None, x1  # Return None for slope and x-intercept as x = constant.

    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1

    return slope, y_intercept

def find_intersection_point(line1, line2):
    slope1, y_intercept1 = find_slope_and_intercept(line1[0], line1[1])
    slope2, y_intercept2 = find_slope_and_intercept(line2[0], line2[1])

    if slope1 is None and slope2 is None:
        # Both lines are vertical and parallel, no unique intersection point.
        return None
    elif slope1 is None:
        # First line is vertical, find x-coordinate from x = constant.
        x_intersection = y_intercept1
        y_intersection = slope2 * x_intersection + y_intercept2
    elif slope2 is None:
        # Second line is vertical, find x-coordinate from x = constant.
        x_intersection = y_intercept2
        y_intersection = slope1 * x_intersection + y_intercept1
    else:
        # Neither line is vertical, find intersection point as usual.
        if slope1 == slope2:
            return None  # The lines are parallel and will never intersect.

        x_intersection = (y_intercept2 - y_intercept1) / (slope1 - slope2)
        y_intersection = slope1 * x_intersection + y_intercept1

    return [x_intersection, y_intersection]

def qr_bbox(bin_img, out_img, qr_bbox_triplets):
  unordered_bboxes = []
  for triplet in qr_bbox_triplets:
    line1 = [triplet[1], triplet[3]]
    line2 = [triplet[2], triplet[4]]
    fourth_point = find_intersection_point(line1, line2)
    cv2.circle(out_img, [int(x) for x in fourth_point], 7, (0, 255, 255), -1)
    unordered_bboxes.append([*triplet[:3], fourth_point])

  final_bboxes = []
  for bbox in unordered_bboxes:
    p1, p2, p3, p4 = bbox
    center = [sum(xs)/4 for xs in zip(*bbox)]
    cv2.circle(out_img, [int(x) for x in center], 7, (255, 0, 255), -1)
    angles = []
    for pt in bbox:
      dir = [(p - c) for p, c in zip(pt, center)]
      angles.append((np.arctan2(dir[0], dir[1]), pt))
    ordered_bbox = np.array([x[1] for x in sorted(angles, key=lambda x: x[0])])
    final_bboxes.append(ordered_bbox)
  
  return final_bboxes, out_img
