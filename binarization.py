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

# Binarization Methods
def binarization_suvola(img, ws=25, **kwargs):
  thresh_sauvola = threshold_sauvola(img, window_size=ws)
  binary_img = img > thresh_sauvola
  return binary_img.astype(np.float32)

def binarization_niblack(img, ws=25, k=0.8, **kwargs):
  thresh_niblack = threshold_niblack(img, window_size=ws, k=0.8)
  binary_img = img > thresh_niblack
  return binary_img.astype(np.float32)

def binarization_otsu(img, **kwargs):
  binary_img = img > threshold_otsu(img)
  return binary_img.astype(np.float32)