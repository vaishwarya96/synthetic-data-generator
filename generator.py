#!/usr/bin/env python
# coding: utf-8



import cv2
import imutils
import numpy as np
import cupy as cp
import math
from os import listdir
from os.path import isfile, join
import io
from IPython.display import display
from PIL import Image
from skimage.exposure import match_histograms
from cupyx.scipy import ndimage
import matplotlib.pyplot as plt
import random

z_step_counter = 0

from utils import *
from data_utils import * 
from image_utils import *

SAVE_TO_TMP = False




def pick_images(images_dict, n_range=[9,12]):
    # Pick random images and add them to list
    n = np.random.randint(n_range[0], n_range[1])
    rand_images = []
    for i in range(n):
        # choosing random images
        rand_key = np.random.choice(list(images_dict.keys()))
        rand_indice = np.random.randint(len(images_dict[rand_key]))
        #rand_image_path = images_dict[rand_key][rand_indice]
        # loading image
        #image_path = join(rand_image_path[0], rand_image_path[1])
        #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if SAVE_TO_TMP: 
            tmp_path = os.path.join(OUTPUT_TMP, "thumbnails/"+rand_key+str(i)+".png")
            check_dirs(tmp_path)
            cv2.imwrite(tmp_path, img)
        #img = match_histograms(img, ref, multichannel=False).astype("uint8")
        #img = cp.asarray(img)
        rand_images.append({"data":images_dict[rand_key][rand_indice], "taxon":rand_key})
    return rand_images




def patchwork(tmp_images, simple_angles = True, size_px = 512, overlapping=0, starter=None, scale=[5,5]):
    art_img = (cp.ones((size_px, size_px))).astype(np.uint8)
    if starter is None:
        global_patch = cp.zeros_like(art_img)
        global_patch_mask_rogn = cp.zeros_like(art_img)
        individual_patches = []
    else:
        global_patch = starter[0]
        global_patch_mask_rogn = starter[1]
        individual_patches = starter[2]
    annotations = []
    global z_step_counter
    for img_obj in tmp_images:
        w, h = img_obj["data"].shape
        #rand_scale = np.random.randint(10, 20)/100
        rand_scale = (scale[0]+np.random.exponential(scale[1]))/100
        weight = 0.95*np.min([w,h])+0.05*np.max([w,h])
        ratio = (rand_scale*size_px)/weight + 0.1
        #ratio = 1.0
        #ratio = random.uniform(0.2,1.0)
        #ratio = random.random() + 0.1
        #ratio = (rand_scale*size_px)/(0.01*w*h)
        img = ndimage.zoom(img_obj["data"], ratio)
        w, h = img.shape
        mask = cp.ones_like(img)*255
        mask_rogn = cp.asarray(round_rectangle(np.min([w,h])//2, w, h, value=255).astype("uint8"))
        # Flipping
        
        if np.random.random()<0.5:
            img = cp.flip(img, 0)
        if np.random.random()<0.5:
            img = cp.flip(img, 1)
        med = np.median(cp.asnumpy(img))
        #img = (np.random.uniform(0.4,1)*(img-med)+med).astype("uint8")
        #PLACING THE IMAGE WITHOUT OVERLAPPING
        #overlap_test = float("inf")
        overlapping_test = True
        n_stop = 200
        area = w*h
        #while overlap_test>area*overlapping and n_stop != 0:
        while overlapping_test and n_stop != 0:
            # Rotating
        
            angle1 = np.random.randint(0,90)
            angle2 = np.random.randint(270,360)
            out = np.stack((angle1,angle2))
            angle = np.random.choice(out)


            rotated = ndimage.rotate(img, angle, axes=(1, 0), reshape=True, output=None, order=0, mode='constant', cval=0.0, prefilter=True)
            rotated_mask_rogn = ndimage.rotate(mask_rogn, angle, axes=(1, 0), reshape=True, output=None, order=0, mode='constant', cval=0.0, prefilter=True)
            # TRANSLATING
            px, py = int(rotated.shape[0]/2), int(rotated.shape[1]/2)
            x, y = np.random.randint(0,size_px-1), np.random.randint(0,size_px-1)
            xmin, xmax, ymin, ymax = x-px, x+px, y-py, y+py
            dxmin, dxmax = (0, -xmin)[xmin<0], (0, size_px-1-xmax)[xmax>size_px-1]
            dymin, dymax = (0, -ymin)[ymin<0], (0, size_px-1-ymax)[ymax>size_px-1]

           # PLACING ON TEMPORARY PATCH/MASL
            patch = cp.zeros_like(art_img)
            patch_mask = cp.zeros_like(art_img)
            patch_mask_rogn = cp.zeros_like(art_img)
            patch[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated[dxmin:2*px+dxmax, dymin:2*py+dymax]
            #patch_mask[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated_mask[dxmin:2*px+dxmax, dymin:2*py+dymax]
            patch_mask_rogn[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated_mask_rogn[dxmin:2*px+dxmax, dymin:2*py+dymax]
            # Testing if there is overlapping by comparing to global mask
            overlapping_test=False
            n_pixels_patch = len(cp.nonzero(patch_mask_rogn)[0])
            n_original_patch = len(cp.nonzero(rotated_mask_rogn)[0])
            if n_pixels_patch<0.01*n_original_patch: # checking if diatom collapses too much with image's border
                overlapping_test=True
            else:
                for prev_patch in individual_patches:
                    prev_patch_mask = prev_patch["patch_mask"]
                    n_overlapping_pixels = len(cp.nonzero(cp.logical_and(patch_mask_rogn, prev_patch_mask))[0])
                    n_pixels_prev_patch = len(cp.nonzero(prev_patch_mask)[0])

                    if n_overlapping_pixels>overlapping*n_pixels_prev_patch or n_overlapping_pixels>overlapping*n_pixels_patch:
                        overlapping_test=True
                        break;
            n_stop -= 1
        if n_stop > 0:
            individual_patches.append({
                "patch": patch.copy(), 
                "patch_mask": patch_mask_rogn.copy(), 
                "center": (int(math.ceil((np.mean([ymin+dymin, ymax+dymax])))), int(math.ceil((np.mean([xmin+dxmin, xmax+dxmax])))))})
            global_patch += patch
            global_patch_mask_rogn += patch_mask_rogn
            #patch_mask_rogn[patch_mask_rogn>0]=1
            annotations.append({
                "taxon": img_obj["taxon"],
                "ymin": xmin+dxmin,
                "xmin": ymin+dymin,
                "ymax": xmax+dxmax,
                "xmax": ymax+dymax,
                "angle": angle,
                "w": w,
                "h": h,
                "patch_mask": patch_mask_rogn,
                "z_index": z_step_counter
                
            })
            

            z_step_counter+=1
        else:
            pass
            #print("Image eliminated...")
    #CREATING FINAL IMAGE
    art_img += global_patch
    #cp.copyto(global_patch, global_patch_mask_rogn, art_img)
    if SAVE_TO_TMP: 
        tmp_patch = cv2.cvtColor(global_patch,cv2.COLOR_GRAY2RGB)
        conts, h = cv2.findContours(global_patch_mask_rogn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tmp_patch = cv2.drawContours(tmp_patch, conts, -1, (0,0,255), 3)
        cv2.imwrite(os.path.join(OUTPUT_TMP, "global_patch.png"), tmp_patch)
    return global_patch, global_patch_mask_rogn, annotations, individual_patches



def fast_img_filling(global_patch, global_patch_mask_rogn, individual_patches, sigma=10e3, verbose=False):
    # IMPROVED img_filling USING CUPY
    final_img = global_patch.copy()
    acc, accw = cp.zeros_like(final_img).astype(np.float32), cp.zeros_like(final_img).astype(np.float32)
    # Finding contours
    conts, h = cv2.findContours(cp.asnumpy(global_patch_mask_rogn), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Getting indices
    height=global_patch.shape[0]
    width=global_patch.shape[1]
    if not hasattr(fast_img_filling, 'W'):
        indices = np.indices((height*2,width*2))
        xMap = cp.asarray(indices[0])
        yMap = cp.asarray(indices[1])
        d2 = cp.square(xMap - width) + cp.square(yMap - height)
        fast_img_filling.W = cp.exp(-d2/sigma)
        fast_img_filling.W[fast_img_filling.W<1e-10] = 1e-10
    # Looping
    i = 0
    known = np.concatenate(conts)
    for kp in known:
        # Counter
        if verbose and i%2000==0:
            print(i, "/", len(known))
            i += 1
        # Init
        xkp, ykp = kp[0][1], kp[0][0]
        val = final_img[xkp, ykp]
        # FILLING
        w = fast_img_filling.W[height-ykp:2*height-ykp,width-xkp:2*width-xkp]
        acc += w*val
        accw += w
    acc = cp.divide(acc, accw)
    acc_img = acc.astype(np.uint8)
    
    # sticking the diatoms
    tmp_global_patch_mask_null = global_patch_mask_rogn==0
    dst = expand(cp.asnumpy(acc_img))
    #pad = 50
    #dst = np.pad(dst, ((pad, pad), (pad, pad), (0, 0)), 'reflect')
    if SAVE_TO_TMP: cv2.imwrite(os.path.join(OUTPUT_TMP, "gradient.png"), dst)
    center = (height//2, width//2)
    np.random.shuffle(individual_patches)
    for patch in individual_patches:
        ks = np.floor(abs(np.random.normal(0, 3, 1))).astype("uint8")[0]*2+1
        patch_img = cp.asnumpy(patch["patch"])
        patch_img = cv2.GaussianBlur(patch_img,(ks,ks),0)
        src = expand(patch_img)
        mask = cp.asnumpy(patch["patch_mask"])
        #center = tuple(x+pad for x in patch["center"])
        center = patch["center"]
        try:
            dst = cv2.seamlessClone(src, dst, mask, center, cv2.MONOCHROME_TRANSFER)
        except ValueError:
            print("hey")
    #dst = dst[pad:-pad, pad:-pad, :]
    return dst




def main_generator(images, simple_angles = False, size_px = 512, verbose=False, overlapping=0, n_diatoms=[9,12], scale_diatoms=[10,10], n_dust=[15,30], scale_dust=[5,10]):    
    padding = 8

    # Placing them randomly on the artboard
    global z_step_counter
    z_step_counter = 0
    images_dict_diatoms, images_dict_debris = images
    
    # Diatoms
    rand_images_diatoms = pick_images(images_dict_diatoms, n_diatoms)
    global_patch, global_patch_mask_rogn, annotations, individual_patches = patchwork(rand_images_diatoms, 
                                                                  simple_angles=simple_angles, 
                                                                  size_px=size_px+2*padding, 
                                                                  overlapping=overlapping,
                                                                  scale=scale_diatoms)
    # Debris
    if not n_dust is None: 
        rand_images_debris = pick_images(images_dict_debris, n_dust)
        global_patch, global_patch_mask_rogn, annotations_dust, individual_patches = patchwork(rand_images_debris, 
                                                  simple_angles=simple_angles, 
                                                  size_px=size_px+2*padding, 
                                                  overlapping=0.25, 
                                                  starter=[global_patch, global_patch_mask_rogn, individual_patches],
                                                  scale=scale_dust)
    
    # Filling the gaps
    final_image = fast_img_filling(global_patch, global_patch_mask_rogn, individual_patches, sigma=10e3, verbose=verbose)
    final_image = final_image[padding:-padding, padding:-padding, :]
    
    to_del = []
    for annotation in annotations:
        for key in ["xmin", "ymin", "xmax", "ymax"]:
            annotation[key] -= padding
            if annotation[key]<0: annotation[key]=0
            if annotation[key]>size_px: annotation[key]=size_px
        if annotation["ymax"]<=0 or annotation["xmax"]<=0 or annotation["xmin"]>=512 or annotation["ymin"]>=512:
            to_del.append(annotation)
    for el in to_del: annotations.remove(el)
    
    if verbose:
        print("Finished!")
        display(Image.fromarray(np.hstack([cp.asnumpy(global_patch), cp.asnumpy(global_patch_mask_rogn)])))
        img_bb = final_image.copy()
        color = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        for annotation in annotations:
            img_bb = cv2.rectangle(img_bb, (annotation["xmin"], annotation["ymin"]), (annotation["xmax"], annotation["ymax"]), color, 4)
            img_bb = cv2.putText(img_bb,  annotation["taxon"], (annotation["xmin"], annotation["ymin"]), font, fontScale, color, 6, cv2.LINE_AA) 
        display(Image.fromarray(img_bb))
    if SAVE_TO_TMP: cv2.imwrite(os.path.join(OUTPUT_TMP, "final_img.png"), final_image)
    return final_image, annotations






