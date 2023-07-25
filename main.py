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

from binarization import *
from finder_grouping import *
from finder_localization import *
from preprocess import *
from qr_bbox import *
from qr_triplet import *
from temporal_stability import *

def process_image(inp_img, preprocess_module, binarization_module, finder_localization_module, finder_pattern_grouper_module, qr_bbox_triplet_module, qr_bbox_module):
  # Pre-process Image
  out_img = preprocess_module(inp_img)

  # Binarize Input Image
  out_img = binarization_module(out_img)

  # Construct the out_matrix
  img_int = (out_img * 255).astype(np.uint8)
  output = np.zeros((img_int.shape[0], img_int.shape[1], 4), dtype=np.uint8)
  output[:, :, 0] = img_int
  output[:, :, 1] = img_int
  output[:, :, 2] = img_int
  output[:, :, 3] = img_int

  # Mark Finder Locations
  finder_locations, debug_img1 = finder_localization_module(out_img)

  if len(finder_locations) < 3: return None, output

  # Group Finder Locations
  grouped_finder_patterns, debug_img2 = finder_pattern_grouper_module(debug_img1, finder_locations)

  if len(grouped_finder_patterns) == 0: return None, output

  # Construct 3-point QR bounding boxes
  qr_bbox_triplets, debug_img3 = qr_bbox_triplet_module(debug_img2, grouped_finder_patterns, finder_locations)

  # Find 4th point of QR bounding box
  qr_bbox_out, debug_img4 = qr_bbox_module(out_img, debug_img3, qr_bbox_triplets)

  # Draw out the final QR boxes
#   for bbox in qr_bbox_out:
#     cv2.polylines(output, [bbox.astype(np.int32)], True, (0, 255, 0, 255), 4)

  return qr_bbox_out, output

# Config for the system
vid = cv2.VideoCapture(0)
pp_func = partial(preprocess_image, img_res_limit=700)
bin_func = partial(binarization_suvola, ws=85)
finder_func = partial(finder_localization_centroid, aspect_filter=0.3, area_ratio_threshold=(1, 4), centroid_closeness=4)
grouper_func = partial(group_finder_locations, side_tol=0.2, hypot_tol=0.15)
trip_bbox_func = partial(qr_bbox_triplet)
qr_bbox_func = partial(qr_bbox)

# Run the event loop
state_dim = (2 + 2) * 4 # bbox coords, velocities
measurement_dim = 2 * 4 # bbox coords
kalman_filter = initialize_kalman_filter(state_dim, measurement_dim)
step = 0
while (True):
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bboxes, bbox_img = process_image(gray, pp_func, bin_func, finder_func, grouper_func, trip_bbox_func, qr_bbox_func)
    
    if step == 0:
        if bboxes is not None and bboxes[0] is not None:
            kalman_filter.statePost[:measurement_dim] = bboxes[0].reshape(measurement_dim, 1)[:measurement_dim].astype(np.float32)
            cv2.polylines(bbox_img, [bboxes[0].astype(np.int32)], True, (0, 255, 0, 255), 4)
    else:
        kalman_filter.predict()
        if bboxes is not None and bboxes[0] is not None:
            kalman_filter.correct(bboxes[0].reshape(measurement_dim, 1)[:measurement_dim].astype(np.float32))

            smoothed_bounding_box = kalman_filter.statePost[:measurement_dim].reshape(-1, 2)
            cv2.polylines(bbox_img, [smoothed_bounding_box.astype(np.int32)], True, (0, 255, 0, 255), 4)
        else:
            smoothed_bounding_box = kalman_filter.statePost[:measurement_dim].reshape(-1, 2)
            cv2.polylines(bbox_img, [smoothed_bounding_box.astype(np.int32)], True, (0, 255, 0, 255), 4)
            
    if bboxes is not None and bboxes[0] is not None:
        cv2.polylines(bbox_img, [bboxes[0].astype(np.int32)], True, (255, 0, 0, 255), 4)
    
    cv2.imshow('frame', bbox_img)
    
    step += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()