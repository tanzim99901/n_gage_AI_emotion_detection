import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext
import numpy as np
    
def getInfo(dataset, labels):

    df_orig = pd.read_csv(dataset)
    
    df_orig = df_orig.dropna()
    
    df_orig.to_csv('Text_dataset' + '\\' + 'Text_Dataset_LATEST.csv')
    
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

for file in os.listdir(dataset_folder):
    # check only text files
    if file.endswith('_LATEST.csv'):
        dataset = dataset_folder + "\\" + file
        
df_orig = pd.read_csv(dataset)

df_orig = df_orig.dropna()

df_orig.to_csv(dataset, index=False)
