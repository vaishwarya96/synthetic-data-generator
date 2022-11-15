#!/usr/bin/env python
# coding: utf-8



import math
import numpy as np
import cv2


def convert_to_square(image, new_size=None, padding = 1):
    #kernel = np.ones((3, 3), np.uint8)
    #image_b = cv2.dilate(image, kernel, iterations = 2)
    crop=1
    image=image[crop:-crop, crop:-crop]
    # Preprocessing
    if not new_size is None:
        ratio = new_size/np.max(image.shape)
        image = cv2.resize(image, 
                           dsize=(math.floor(ratio*image.shape[1])-2*padding, math.floor(ratio*image.shape[0])-2*padding), 
                           interpolation=cv2.INTER_LINEAR)

    # Converting to square
    square_size = np.max(image.shape)
    h, w = image.shape[0], image.shape[1]
    delta_w, delta_h = square_size - w, square_size - h
    left, top = delta_w//2, delta_h//2    
    blur_size = int(np.max(image.shape)/4)*2+1
    blured_image=cv2.GaussianBlur(image,(blur_size,blur_size),0)
    square_image_blurred = cv2.copyMakeBorder(blured_image, top+padding, delta_h-top+padding, left+padding, delta_w-left+padding, cv2.BORDER_REPLICATE)
    square_image = square_image_blurred.copy()
    square_image[top+padding:top+h+padding, left+padding:left+w+padding] = image.copy()

    # Seamless cloning
    height=square_image_blurred.shape[0]
    width=square_image_blurred.shape[1]
    mask_ref=np.zeros_like(square_image).astype("uint8")
    mask_ref[top+1:top+h+1, left+1:left+w+1] = 255
    center = (height//2, width//2)
    src = expand(square_image)
    dst = expand(square_image_blurred)
    final_image = cv2.seamlessClone(src, dst, mask_ref, center, cv2.NORMAL_CLONE)

    if not new_size is None: 
            final_image = cv2.resize(final_image, 
                       dsize=(new_size, new_size), 
                       interpolation=cv2.INTER_CUBIC)
    return final_image

def expand(image):
    if image.ndim==2:
         image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image


# In[ ]:




