#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
import errno
from skimage.exposure import match_histograms
from sklearn.model_selection import train_test_split
import cupy as cp

from data_utils import *

def list_images(data_path, diatoms=True, taxon_filter=None, ref=None):
    # List all images in dataset and putting them in a dict {"key": "imgpath"}
    i = 0
    images_path = []
    for path in data_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split(".")[1] in ["png", "tif", "tiff", "TIF", "PNG","jpg","JPG"]:
                    images_path.append((root, file))
                    i+=1
    #for path in data_path: images_path.extend([(path, f) for f in listdir(path) if isfile(join(path, f))])
    #print(images_path)
    images_dict = {}
    if not taxon_filter is None: selected_taxons = get_selected_taxons(taxon_filter)
    for image_path in images_path:
        if len(image_path[1].split('.'))>=2: 
            # defining key
            if diatoms: 
                if taxon_filter is None:
                    key="diatom"
                else:
                    key = image_path[1].split('_')[1]
                    if not key in selected_taxons:
                        key = None
            else: 
                key="mono"
            if not key is None:
                image_path = join(image_path[0], image_path[1])
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if not ref is None:
                    img = match_histograms(img, ref, multichannel=False).astype("uint8")
                img = cp.asarray(img)
                images_dict.setdefault(key,[]).append(img)
                #images_dict.setdefault(key,[]).append(image_path)
    #print(i,"images found!")
    return images_dict   

def split_img_list(dict_input, perc):
    train = {}
    val = {}
    for key in list(dict_input.keys()):
        images = dict_input[key].copy()
        np.random.shuffle(images)
        train[key], val[key] = train_test_split(images, test_size=perc, random_state=42)
    return train, val

def round_rectangle(radius, w, h, value=255):
    thickness = -1
    color = value
    rr = np.zeros((w, h))
    rr = cv2.circle(rr, (radius, radius), radius, color, thickness) 
    rr = cv2.circle(rr, (h-radius, radius), radius, color, thickness) 
    rr = cv2.circle(rr, (radius, w-radius), radius, color, thickness) 
    rr = cv2.circle(rr, (h-radius, w-radius), radius, color, thickness) 
    rr = cv2.rectangle(rr, (0, radius), (h, w-radius), color, thickness) 
    rr = cv2.rectangle(rr, (radius, 0), (h-radius, w), color, thickness) 
    return rr

def resize_img(img, scale_percent):
    if scale_percent==1:
        return img
    else:
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height) 
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        return resized
    
def cp_median(x, axis):
    xp = cp.get_array_module(x)
    n = x.shape[axis]
    s = xp.sort(x, axis)
    m_odd = xp.take(s, n // 2, axis)
    if n % 2 == 1:  # 奇数個
        return m_odd
    else:
        m_even = xp.take(s, n // 2 - 1, axis)
        return (m_odd + m_even) / 2


# In[ ]:




