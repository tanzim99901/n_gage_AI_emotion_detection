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
import librosa
import soundfile as sf
import colorednoise as cn

from pydub import AudioSegment

import random

dataset_folder = 'Audio processing' + '//' + 'audioData'

# list to store files
res = []
# Iterate directory
for file in os.listdir(dataset_folder):
    # check only text files
    if 'aug' in file:
        res.append(file)

total = len(res)
k = 0
print("Removing augmented audio clips...")
print(total)

for i in range(0,total):
    percent = ((i+1)/(total)) * 100
    print("Removing augmented audio clip " + str(i+1) + " / " + str(total) + " (" + str(round(percent,2)) + " %)")
    os.remove(dataset_folder + "\\" + res[i])