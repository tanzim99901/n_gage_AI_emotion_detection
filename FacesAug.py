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

df_orig = pd.read_csv(faceLabel_folder + "\\faceLabels.csv")

mask = df_orig.FileName.str.contains("aug")
df_aug = df_orig[mask]
df = df_orig[~mask]

df_temp = pd.DataFrame(columns = ['FileName', 'Label'])

df_cat1 = df[df['Label'] == 1]

df_cat1 = df_cat1.reset_index(drop=True)

df_cat2 = df[df['Label'] == 2]

df_cat2 = df_cat2.reset_index(drop=True)

res = []

# Iterate directory
for file in os.listdir(face_folder):
    # check only text files
    if file.endswith('.jpg'):
        res.append(file)


## For category 2, use 5 augmentations, horFlip, imgCrop, addNoise, contrast sigmoid, contrast linear
## For category 1, use 10 augmentations        
total = df_cat1.shape[0] + df_cat2.shape[0]
print("Augmenting Category 1 images...")
tot_cat1 = df_cat1.shape[0]
for i in range(0,df_cat1.shape[0]):
    if df_cat1.loc[i,'FileName'] in res:
        percent = ((i+1)/(tot_cat1)) * 100
        tot_percent = ((i+1)/(total)) * 100
        print("Augmenting Category 1 Image " + str(i+1) + " / " + str(tot_cat1) + " (" + str(percent) + " %)" + " \t " + "Overall: " + str(tot_percent) + " %")
        img = imageio.imread(face_folder + "\\" + df_cat1.loc[i,'FileName'])
        
        horImg = horFlip(img)
        rotImg = imgRot(img)
        cropImg = imgCrop(img)
        noiseImg = addNoise(img)
        shearImg = imgShear(img)
        contGammaImg = contrast(img,"gamma")
        contSigmoidImg = contrast(img,"sigmoid")
        contLinearImg = contrast(img,"linear")
        transElasticImg = trans(img,"elastic")
        transJigsawImg = trans(img,"jigsaw")
        
        temp_filename = df_cat1.loc[i,'FileName'].replace('.jpg','')
        #print(temp_filename)
        
        cv2.imwrite(face_folder + "\\" + temp_filename + "_1_aug.jpg", horImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_2_aug.jpg", rotImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_3_aug.jpg", cropImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_4_aug.jpg", noiseImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_5_aug.jpg", shearImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_6_aug.jpg", contGammaImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_7_aug.jpg", contSigmoidImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_8_aug.jpg", contLinearImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_9_aug.jpg", transElasticImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_10_aug.jpg", transJigsawImg)
        
        df1 = pd.DataFrame({"FileName":[temp_filename + "_1_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_2_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_3_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_4_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_5_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_6_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_7_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_8_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_9_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_10_aug.jpg"],"Label":[1]})
        df_temp = df_temp.append(df1, ignore_index = True)

## For category 2, use 5 augmentations, horFlip, imgCrop, addNoise, contrast sigmoid, contrast linear
print("Augmenting Category 2 images...")
tot_cat2 = df_cat2.shape[0]
for i in range(0,df_cat2.shape[0]):
    if df_cat2.loc[i,'FileName'] in res:
        percent = ((i+1)/(tot_cat2)) * 100
        tot_percent = ((i+1+tot_cat1)/(total)) * 100
        print("Augmenting Category 2 Image " + str(i+1) + " / " + str(tot_cat2) + " (" + str(percent) + " %)" + " \t " + "Overall: " + str(tot_percent) + " %")
        img = imageio.imread(face_folder + "\\" + df_cat2.loc[i,'FileName'])
        
        horImg = horFlip(img)
        cropImg = imgCrop(img)
        noiseImg = addNoise(img)
        contSigmoidImg = contrast(img,"sigmoid")
        contLinearImg = contrast(img,"linear")
        
        temp_filename = df_cat2.loc[i,'FileName'].replace('.jpg','')
        #print(temp_filename)
        
        cv2.imwrite(face_folder + "\\" + temp_filename + "_1_aug.jpg", horImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_2_aug.jpg", cropImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_3_aug.jpg", noiseImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_4_aug.jpg", contSigmoidImg)
        cv2.imwrite(face_folder + "\\" + temp_filename + "_5_aug.jpg", contLinearImg)
        
        df1 = pd.DataFrame({"FileName":[temp_filename + "_1_aug.jpg"],"Label":[2]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_2_aug.jpg"],"Label":[2]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_3_aug.jpg"],"Label":[2]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_4_aug.jpg"],"Label":[2]})
        df_temp = df_temp.append(df1, ignore_index = True)
        df1 = pd.DataFrame({"FileName":[temp_filename + "_5_aug.jpg"],"Label":[2]})
        df_temp = df_temp.append(df1, ignore_index = True)

df = df.append(df_temp, ignore_index = True)

df.to_csv(faceLabel_folder + "\\" + "faceLabels.csv", index=None)