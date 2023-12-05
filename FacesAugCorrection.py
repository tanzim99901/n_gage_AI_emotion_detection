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

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list
    
video_folder = "videos"
face_folder = "faceData"
faceLabel_folder = "faceDataLabels"

df = pd.read_csv(faceLabel_folder + "\\faceLabels.csv")
#df = pd.read_csv(faceLabel_folder + "\\augCorrectionTest.csv")

df_fileName_string = create_list_empty_strings(df.shape[0])
df_label_string = create_list_empty_strings(df.shape[0])

df_temp = pd.DataFrame(columns = ['FileName', 'Label'])

mask = df.FileName.str.contains("aug")
df_aug = df[mask]
df_nonAug = df[~mask]

df_nonAug = df_nonAug.reset_index(drop=True)
df_aug = df_aug.reset_index(drop=True)

print("Total images: " + str(df.shape[0]))
print("Total augmented images: " + str(df_aug.shape[0]))
print("Total non-augmented images: " + str(df_nonAug.shape[0]))
    
count = 0

#print(df_aug.loc[0,'FileName'])

for i in range(0,df_aug.shape[0]):
    percent = ((i+1)/(df_aug.shape[0])) * 100
    print("Correcting augmented Image " + str(i+1) + " / " + str(df_aug.shape[0]) + " (" + str(round(percent,5)) + " %)")
    if '_1_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_1_aug.jpg','') + '.jpg'
        suffix = '_1_aug.jpg'
    elif '_2_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_2_aug.jpg','') + '.jpg'
        suffix = '_2_aug.jpg'
    elif '_3_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_3_aug.jpg','') + '.jpg'
        suffix = '_3_aug.jpg'
    elif '_4_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_4_aug.jpg','') + '.jpg'
        suffix = '_4_aug.jpg'
    elif '_5_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_5_aug.jpg','') + '.jpg'
        suffix = '_5_aug.jpg'
    elif '_6_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_6_aug.jpg','') + '.jpg'
        suffix = '_6_aug.jpg'
    elif '_7_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_7_aug.jpg','') + '.jpg'
        suffix = '_7_aug.jpg'
    elif '_8_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_8_aug.jpg','') + '.jpg'
        suffix = '_8_aug.jpg'
    elif '_9_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_9_aug.jpg','') + '.jpg'
        suffix = '_9_aug.jpg'
    elif '_10_aug.jpg' in df_aug.loc[i,'FileName']:
        orig_fileName = df_aug.loc[i,'FileName'].replace('_10_aug.jpg','') + '.jpg'
        suffix = '_10_aug.jpg'
    
    
    df1 = df_nonAug[df_nonAug['FileName'] == orig_fileName]
    df1 = df1.reset_index(drop=True)
    df1.loc[0,'FileName'] = df1.loc[0,'FileName'].replace('.jpg',suffix)
    df_temp = df_temp.append(df1, ignore_index = True)
        
df_final = df_nonAug.append(df_temp, ignore_index = True)
df_final.to_csv(faceLabel_folder + "\\" + "faceLabelsCorrected.csv", index=None)