import pandas as pd
import os
import numpy as np
import wave
import librosa
from python_speech_features import *
import sys
import pickle

def getInfo(dataset_folder, labels, ignoreAug):
    res = []
    counter = np.zeros(len(labels), dtype=int) 
    
    if ignoreAug == False:
        for file in os.listdir(dataset_folder):
            if file.endswith('.wav'):
                res.append(file)
    
    else:
        for file in os.listdir(dataset_folder):
            if file.endswith('.wav'):
                if 'aug' not in file:
                    res.append(file)
    
    for i in range(0,len(res)):
        filename = res[i].replace('.wav','')
        label = int(filename[-1])
        
        for j in range(0, len(labels)):
            if label == labels[j]:
                counter[j] = counter[j] + 1
    total = len(res)
    return counter, total
        
dataset_folder = 'Audio processing' + '//' + 'audioData'
#dataset_folder = 'Audio processing (WELDER)' + '//' + 'audioData'
#features_folder = dataset_folder + '//' + 'audioFeatures'

labels = [0,1,2,3,4]
class_names = ['Neutral', 'Negative-deactivated', 'Positive-deactivated', 'Positive-activated', 'Negative-activated']

counter_aug = np.zeros(len(labels), dtype=int) 
counter_nonAug = np.zeros(len(labels), dtype=int)

counter_nonAug, total_nonAug = getInfo(dataset_folder, labels, ignoreAug = True)
counter_aug, total_aug = getInfo(dataset_folder, labels, ignoreAug = False)

print("Before augmentation")
print("===================")

for i in range(0, len(labels)):
    
    print("Count of category " + str(labels[i]) + " (" + class_names[i] + ") is " + str(counter_nonAug[i]))
    
print("Total sample count is " + str(total_nonAug))

print(" ")
print("After augmentation")
print("===================")

for i in range(0, len(labels)):
    
    print("Count of category " + str(labels[i]) + " (" + class_names[i] + ") is " + str(counter_aug[i]))
    
print("Total sample count is " + str(total_aug))