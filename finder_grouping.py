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

def group_finder_locations(out_img, finder_descs, side_tol=0.05, hypot_tol=0.05):
  centroids = [(x[0][0], x[0][1]) for x in finder_descs]
  distance_matrix = cdist(centroids, centroids)

  pairings = []

  for i in range(len(centroids)):
    for j in range(i + 1, len(centroids)):
      for k in range(j + 1, len(centroids)):
        if i == j or j == k or i == k: continue

        sides = [
          distance_matrix[i][j],
          distance_matrix[j][k],
          distance_matrix[i][k]
        ]

        max_idx = 0
        sides_idx = [1, 2]
        if sides[1] > sides[max_idx]:
          max_idx = 1
          sides_idx = [0, 2]
        if sides[2] > sides[max_idx]:
          max_idx = 2
          sides_idx = [0, 1]

        hypot = sides[max_idx]
        sideA, sideB = [sides[i] for i in sides_idx]
        calc_hypot = math.sqrt((sideA**2) + (sideB**2))

        # print(hypot, sideA, sideB, abs(sideA - sideB), (side_tol * max(sideA, sideB)), abs(math.sqrt((sideA**2) + (sideB**2)) - hypot), (hypot_tol * max(calc_hypot, hypot)))

        if abs(sideA - sideB) >= (side_tol * max(sideA, sideB)): continue

        if abs(calc_hypot - hypot) >= (hypot_tol * max(calc_hypot, hypot)): continue

        side_opp_point_idx = [k, i, j]
        pairings.append((side_opp_point_idx[max_idx], *[side_opp_point_idx[idx] for idx in sides_idx])) # Hypotenuse, Side1, Side 2

        # For debug purposes only
        for idx1, idx2 in  [(i, j), (j, k), (i, k)]:
          cv2.line(out_img, [int(x) for x in centroids[idx1]], [int(x) for x in centroids[idx2]], (0, 0, 255), 3)
  
  return pairings, out_img