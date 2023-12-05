import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext
import numpy as np

   
def getInfo(dataset, labels, ignoreAug):

    df_orig = pd.read_csv(dataset)
    
    if ignoreAug == True:
        mask = df_orig.FileName.str.contains("aug")
        df_aug = df_orig[mask]
        df = df_orig[~mask]
    else:
        df = df_orig
    
    label_df = df['one pred']
    
    counter = np.zeros(len(labels), dtype=int)
    
    total = 0

    for i in range(0,label_df.shape[0]):
        for j in range(0, len(labels)):
            if label_df[i] == labels[j]:
                counter[j] = counter[j] + 1

    total = df.shape[0]
    return counter, total    

        
        
dataset_folder = 'Text_Dataset\\one pred'
#dataset_folder = 'Text_Dataset (WELDER)\\one pred'

for file in os.listdir(dataset_folder):
    # check only text files
    if file.endswith('_LATEST.csv'):
        dataset = dataset_folder + "\\" + file

labels = [0,1,2,3,4]
class_names = ['Neutral', 'Negative-deactivated', 'Positive-deactivated', 'Positive-activated', 'Negative-activated']

counter_aug = np.zeros(len(labels), dtype=int) 
counter_nonAug = np.zeros(len(labels), dtype=int)

if dataset.endswith('_LATEST.csv'):
    counter_nonAug, total_nonAug = getInfo(dataset, labels, ignoreAug = True)
    counter_aug, total_aug = getInfo(dataset, labels, ignoreAug = False)

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
else:
    print("Dataset not found")