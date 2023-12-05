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

def audioAugment(filename):
    global backgroundNoise_folder
    #print(filename)
    filename_1 = filename.replace('.wav','')
    label = int(filename_1[-1])
    
    
        
    original_audio, orig_sr = librosa.load(dataset_folder + "//" + filename, sr=None)
    

    
    ## 1 add white noise
    
    noise_factor = 0.005
    white_noise = np.random.randn(len(original_audio)) * noise_factor
    
    augmented_audio = original_audio + white_noise
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_1_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_1_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
    
    
    
    
    ## 2 add pink noise
    
    pink_noise = cn.powerlaw_psd_gaussian(2.5, len(original_audio))
    
    augmented_audio = original_audio + pink_noise
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_2_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_2_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
 
 
 
 
    ## 3 Speed up audio
    
    rate = random.uniform(1.2, 1.8)
    augmented_audio = librosa.effects.time_stretch(original_audio, rate = rate)
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_3_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_3_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
    
    
    
    ## 4 Slow down audio
    
    rate = random.uniform(0.6, 0.9)
    augmented_audio = librosa.effects.time_stretch(original_audio, rate = rate)
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_4_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_4_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
    
    
    
    ## 5 Pitch up
    
    pitch_factor = random.randint(2, 5)
    augmented_audio = librosa.effects.pitch_shift(original_audio, sr=orig_sr, n_steps=pitch_factor)
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_5_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_5_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
    
    
    
    ## 6 Pitch down
    
    pitch_factor = random.randint(2, 5)
    augmented_audio = librosa.effects.pitch_shift(original_audio, sr=orig_sr, n_steps=-pitch_factor)
    
    out_name = 'jack'
    
    if label == 1:
        out_name = filename.replace('1.wav','') + 'aug_6_1.wav'
        #print(out_name)
    elif label == 2:
        out_name = filename.replace('2.wav','') + 'aug_6_2.wav'
        #print(out_name)
    
    sf.write(dataset_folder + "//" + out_name, augmented_audio, orig_sr)
    
    
    
    
    ##### 3, 4, 5 Background noise section
    
    
    orig_audio = AudioSegment.from_file(dataset_folder + "//" + filename, format="wav")
    
    for i in range(0, len(backgroundNoise)):
        noise_file = backgroundNoise[i]

        bg_noise = noise_file[0:len(orig_audio)]
        augmented_audio = orig_audio.overlay(bg_noise)

        out_name = 'jack'
        
        if label == 1:
            out_name = filename.replace('1.wav','') + 'aug_' + str(i+7) + '_1.wav'
            #print(out_name)
        elif label == 2:
            out_name = filename.replace('2.wav','') + 'aug_' + str(i+7) + '_2.wav'
            #print(out_name)
        
        augmented_audio.export(dataset_folder + "//" + out_name, format="wav")
    

dataset_folder = 'Audio processing' + '//' + 'audioData'
#dataset_folder = 'Audio processing (WELDER)' + '//' + 'audioData'
backgroundNoise_folder = 'Audio processing' + '//' + 'Background_Noise'

# list to store files
res = []
# Iterate directory
for file in os.listdir(dataset_folder):
    # check only text files
    if file.endswith('1.wav'):
        if 'aug' not in file:
            res.append(file)
    if file.endswith('2.wav'):
        if 'aug' not in file:
            res.append(file)

backgroundNoise = []
for file in os.listdir(backgroundNoise_folder):
    # check only text files
    if file.endswith('.wav'):
        #noise, sr = librosa.load(backgroundNoise_folder + "\\" + file)
        noise = AudioSegment.from_file(backgroundNoise_folder + "\\" + file, format="wav")
        backgroundNoise.append(noise)

total = len(res)        
for i in range(0,len(res)):
    percent = ((i+1)/(total)) * 100
    print("Augmenting Audio file " + str(i+1) + " / " + str(total) + " (" + str(percent) + " %)")
    audioAugment(res[i])
    