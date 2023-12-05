import imageio
import imgaug as ia
import imgaug.augmenters as iaa

import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext

import ipyplot

import cv2

import numpy as np
np.bool = np.bool_

def horFlip(input_img):
    #Horizontal Flip
    hflip= iaa.Fliplr(p=1.0)
    mod_img= hflip.augment_image(input_img)
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def imgRot(input_img):
    rot1 = iaa.Affine(rotate=(-50,20))
    mod_img = rot1.augment_image(input_img)
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def imgCrop(input_img):
    crop1 = iaa.Crop(percent=(0, 0.3)) 
    mod_img = crop1.augment_image(input_img)
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def addNoise(input_img):
    noise = iaa.AdditiveGaussianNoise(10,40)
    mod_img = noise.augment_image(input_img)
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def imgShear(input_img):
    shear = iaa.Affine(shear=(-40,40))
    mod_img = shear.augment_image(input_img)
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def contrast(input_img, contrastType):
    contrast_gamma = iaa.GammaContrast((0.5, 2.0))
    contrast_sig = iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6))
    contrast_lin = iaa.LinearContrast((0.6, 0.4))
    
    if contrastType == "gamma":
        mod_img = contrast_gamma.augment_image(input_img)
    elif contrastType == "sigmoid":
        mod_img = contrast_sig.augment_image(input_img)
    elif contrastType == "linear":
        mod_img = contrast_lin.augment_image(input_img)
    
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img

def trans(input_img, transType):

    elastic = iaa.ElasticTransformation(alpha=60.0, sigma=4.0)
    polar = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.2, 0.2)))
    jigsaw = iaa.Jigsaw(nb_rows=20, nb_cols=15, max_steps=(1, 3))
    
    if transType == "elastic":
        mod_img = elastic.augment_image(input_img)
    elif transType == "polar":
        mod_img = polar.augment_image(input_img)
    elif transType == "jigsaw":
        mod_img = jigsaw.augment_image(input_img)
    
    rgb_mod_img = mod_img[:, :, ::-1]
    return rgb_mod_img
    
video_folder = "videos"
face_folder = "faceData"
faceLabel_folder = "faceDataLabels"

df = pd.read_csv(faceLabel_folder + "\\faceLabels.csv")

res = []


# Iterate directory
for file in os.listdir(face_folder):
    # check only text files
    if file.endswith('_aug.jpg'):
        res.append(file)


## For category 2, use 5 augmentations, horFlip, imgCrop, addNoise, contrast sigmoid, contrast linear
## For category 1, use 10 augmentations        
total = len(res)
k = 0
print("Removing augmented images...")
for i in range(0,df.shape[0]):
    if "_aug.jpg" in df.loc[i,'FileName']:
        if df.loc[i,'FileName'] in res:
            percent = ((k+1)/(total)) * 100
            print("Removing augmented Image " + str(k+1) + " / " + str(total) + " (" + str(round(percent,2)) + " %)")
            os.remove(face_folder + "\\" + df.loc[i,'FileName'])
            df = df.drop(axis=0,index=i)
            k = k + 1

df.to_csv(faceLabel_folder + "\\" + "faceLabels.csv", index=None)