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

def qr_bbox_triplet(out_img, finder_triplets, finder_descs):
  def find_unit_vector(pt1, pt2): # from pt2 -> pt1
    diff_vector = [x1 - x2 for x1, x2 in zip(pt1, pt2)]
    mag = math.sqrt(sum(x ** 2 for x in diff_vector))
    unit_vector = [x / mag for x in diff_vector]
    return unit_vector

  sqrt_18 = math.sqrt(18)

  qr_bbox_triplets = []
  for triplet in finder_triplets:
    descs = [finder_descs[i] for i in triplet]
    corner_pt = descs[0]
    side_pt1 = descs[1]
    side_pt2 = descs[2]

    side_pt1_main = [((side_pt1[1] * sqrt_18 * dx) + x) for dx, x in zip(find_unit_vector(side_pt1[0], side_pt2[0]), side_pt1[0])]
    side_pt2_main = [((side_pt2[1] * sqrt_18 * dx) + x) for dx, x in zip(find_unit_vector(side_pt2[0], side_pt1[0]), side_pt2[0])]

    # For corner pt the unit vector is just an addition of the other two unit vectors
    spt1_cpt_unit = find_unit_vector(corner_pt[0], side_pt1[0])
    spt2_cpt_unit = find_unit_vector(corner_pt[0], side_pt2[0])
    disp_vec = [x1 + x2 for (x1, x2) in zip(spt1_cpt_unit, spt2_cpt_unit)]
    disp_mag = math.sqrt(sum(x ** 2 for x in disp_vec))
    disp_unit = [(x / disp_mag) for x in disp_vec]
    corner_pt_main = [((corner_pt[1] * sqrt_18 * dx) + x) for dx, x in zip(disp_unit, corner_pt[0])]

    cv2.circle(out_img, [int(x) for x in side_pt1_main], 7, (0, 255, 255), -1)
    cv2.circle(out_img, [int(x) for x in side_pt2_main], 7, (0, 255, 255), -1)
    cv2.circle(out_img, [int(x) for x in corner_pt_main], 7, (0, 255, 255), -1)

    # We should also mark secondaries so we can adjust for the final bbox.
    side_pt1_second = [((side_pt1[1] * 3 * dx) + x) for dx, x in zip(find_unit_vector(side_pt1[0], corner_pt[0]), side_pt1[0])]
    side_pt2_second = [((side_pt2[1] * 3 * dx) + x) for dx, x in zip(find_unit_vector(side_pt2[0], corner_pt[0]), side_pt2[0])]

    corner_pt_pt1 = [((corner_pt[1] * 3 * dx) + x) for dx, x in zip(spt1_cpt_unit, corner_pt[0])]
    corner_pt_pt2 = [((corner_pt[1] * 3 * dx) + x) for dx, x in zip(spt2_cpt_unit, corner_pt[0])]

    cv2.circle(out_img, [int(x) for x in side_pt1_second], 1, (255, 0, 255), -1)
    cv2.circle(out_img, [int(x) for x in side_pt2_second], 1, (255, 0, 255), -1)
    cv2.circle(out_img, [int(x) for x in corner_pt_pt1], 1, (255, 0, 255), -1)
    cv2.circle(out_img, [int(x) for x in corner_pt_pt2], 1, (255, 0, 255), -1)

    qr_bbox_triplets.append([corner_pt_main, side_pt1_main, side_pt2_main, side_pt1_second, side_pt2_second]) # Corner point, side point 1, side point 2

  return qr_bbox_triplets, out_img