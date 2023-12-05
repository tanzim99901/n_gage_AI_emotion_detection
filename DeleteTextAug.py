import textaugment, gensim

from textaugment import Word2vec
from textaugment import Wordnet
from textaugment import Translate

from textaugment import EDA

import nltk

import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list
    
def deleteAugmentDataset(filename):
    df = pd.read_csv(filename)
    mask = df.FileName.str.contains("aug")
    df_aug = df[mask]
    df_nonAug = df[~mask]
    
    df_nonAug = df_nonAug.reset_index(drop=True)
    df_aug = df_aug.reset_index(drop=True)
    
    print("Total text samples: " + str(df.shape[0]))
    print("Total augmented text samples: " + str(df_aug.shape[0]))
    print("Total non-augmented text samples: " + str(df_nonAug.shape[0]))

    print("Removing augmented text samples...")

    df_nonAug.to_csv(filename, index=None)
    
dataset_dir_path = 'Text_Dataset\\one pred'

deleteAugmentDataset(dataset_dir_path + '\\Text_Dataset_LATEST.csv')