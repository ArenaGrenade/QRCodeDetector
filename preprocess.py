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

def preprocess_image(inp_img, img_res_limit=700):
  h, w = inp_img.shape[:2]
  if h >= img_res_limit or w >= img_res_limit:
    rf = max(h / img_res_limit, w / img_res_limit)
    out_img = cv2.resize(inp_img, (int(w / rf), int(h / rf)))
    return out_img
  return inp_img