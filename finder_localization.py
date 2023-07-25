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

# Finder Localization Methods
def finder_localization_centroid(img, aspect_filter=0.3, area_ratio_threshold=(1, 4), centroid_closeness=4, **kwargs):
  # CC form Opencv needs 8 bit int :(
  img_int = (img * 255).astype(np.uint8)
  output = cv2.connectedComponentsWithStats(255 - img_int, 4,  cv2.CV_32S)
  (numLabels, labelled_img, stats, centroids) = output

  # Output image initialization
  output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  output[:, :, 0] = img_int
  output[:, :, 1] = img_int
  output[:, :, 2] = img_int

  # Filter for valid componnets
  valid_components = []
  for i in range(numLabels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 9: continue

    aspect_ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]
    # Reversed comparision, but just checks if aspect ratio is not in valid range
    if (1 + aspect_filter) < aspect_ratio < (1 - aspect_filter):
      continue

    # TODO: Optimizable using a AVL - maybe could be slower due to the almost ordered inserting.
    bounding_box = [
      stats[i, cv2.CC_STAT_LEFT], # left
      stats[i, cv2.CC_STAT_TOP], # top
      stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH], # right
      stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] # bottom
    ]
    valid_components.append([*centroids[i], area, bounding_box, i]) # c_x, c_y, area, bbox, label

  # Sort valid components. Refer to above TODO for optimizing using AVL / BST
  sorted_components = sorted(valid_components, key = lambda x: (x[0], x[1]))

  # Find Matching Centroids
  # Using adjacent 2-window search, must likely work well. But imo a bucketing
  # followed by full n^2 search within each bucket must work best.
  possible_finder_descriptors = [] # each tuple has possible outline followed by possible inside square
  for i in range(len(sorted_components) - 1):
    # Assign bigger and smaller squares
    if sorted_components[i][2] > sorted_components[i + 1][2]:
      big = i
      small = i + 1
    else:
      big = i + 1
      small = i
    b_desc = sorted_components[big]
    s_desc = sorted_components[small]

    # Check Area Ratio is between (1,4]
    area_ratio = (b_desc[2] / s_desc[2])
    if area_ratio > area_ratio_threshold[1] or area_ratio <= area_ratio_threshold[0]: continue

    # Check bbox of s_desc within b_desc
    if b_desc[3][0] >= s_desc[3][0] or b_desc[3][1] >= s_desc[3][1] or b_desc[3][2] <= s_desc[3][2] or b_desc[3][3] <= s_desc[3][3]:
      continue

    # Check centroid distance is less than 3
    if math.sqrt((b_desc[0] - s_desc[0])**2 + (b_desc[1] - s_desc[1])**2) >= centroid_closeness:
      continue
    
    cv2.rectangle(output, (b_desc[3][0], b_desc[3][1]), (b_desc[3][2], b_desc[3][3]), (255, 0, 0), 5)
    cv2.rectangle(output, (s_desc[3][0], s_desc[3][1]), (s_desc[3][2], s_desc[3][3]), (255, 0, 0), 5)

    possible_finder_descriptors.append((b_desc, s_desc))

  # Calculate moments for each of the possbile finder descriptor pairs and calculate new centroids and module width
  output_finder_descriptors = []
  for b_desc, s_desc in possible_finder_descriptors:
    try:
      b_obj_img = (labelled_img == b_desc[4]).astype(np.uint8) * 255
      s_obj_img = (labelled_img == s_desc[4]).astype(np.uint8) * 255

      b_M = cv2.moments(b_obj_img)
      s_M = cv2.moments(s_obj_img)

      centroid = [
        (b_M["m10"] + s_M["m10"]) / (b_M["m00"] + s_M["m00"]),
        (b_M["m01"] + s_M["m01"]) / (b_M["m00"] + s_M["m00"])
      ]
      
      # module_width = (math.sqrt(b_desc[2]) + math.sqrt(s_desc[2])) / (math.sqrt(24) + math.sqrt(9))
      module_width = math.sqrt((b_desc[2] + s_desc[2]) / 33)
      # Second is faster, but first is more accurate

      cv2.circle(output, [int(x) for x in centroid], 5, (0, 255, 0), -1)

      output_finder_descriptors.append((centroid, module_width, b_desc, s_desc))
    except:
      continue

  return output_finder_descriptors, output