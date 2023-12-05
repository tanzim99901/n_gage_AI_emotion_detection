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
    
def augmentDataset(filename):
    global model
    global t_Word2vec, t_Wordnet, t_Translate, t_EDA, aug_bert
    global len_BERT
    df = pd.read_csv(filename)
    
    df_temp = pd.DataFrame(columns = df.columns)
    
    df_cat1 = df[df['one pred'] == 1]

    df_cat1 = df_cat1.reset_index(drop=True)

    df_cat2 = df[df['one pred'] == 2]

    df_cat2 = df_cat2.reset_index(drop=True)
    
    #text = df.loc[0,'text'].replace("'", "")
    
    total = df_cat1.shape[0] + df_cat2.shape[0]
    
    print("Augmenting Category 1 Text...")
    tot_cat1 = df_cat1.shape[0]
    
    for i in range(0,df_cat1.shape[0]):
        percent = ((i+1)/(tot_cat1)) * 100
        tot_percent = ((i+1)/(total)) * 100
        print("Augmenting Category 1 Text " + str(i+1) + " / " + str(tot_cat1) + " (" + str(percent) + " %)" + " \t " + "Overall: " + str(tot_percent) + " %")
        
        text = df_cat1.loc[i,'text']
        #text = 'Alright, were going to go ahead and assign tasks for everyone.'
        out_Word2vec = t_Word2vec.augment(text)
        out_Wordnet = t_Wordnet.augment(text)
        #out_Translate = t_Translate.augment(text)
        out_Synonym = t_EDA.synonym_replacement(text)
        # out_Deletion = t_EDA.random_deletion(text, p=0.2)
        # out_Swap = t_EDA.random_swap(text)
        # out_Insertion = t_EDA.random_insertion(text)
        
        out_BERT = create_list_empty_strings(len_BERT)
        
        for k in range(0,len_BERT):
            out_BERT[k] = aug_bert.augment(text)
        
        name = df_cat1.loc[i, 'FileName'].replace(".txt","_aug.txt")
        timeframe = df_cat1.loc[i, 'TimeFrame']
        pos = df_cat1.loc[i, 'Positive score (+/-)']
        act = df_cat1.loc[i, 'Active score (+/-)']
        one_pred = df_cat1.loc[i, 'one pred']
        
        textAug = out_Word2vec.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        textAug = out_Wordnet.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        
        textAug = out_Synonym.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        for k in range(0,len(out_BERT)):
            #print(out_BERT[k][0])
            textAug = out_BERT[k][0].replace("'", "")
            textAug = textAug.replace('"', '')
            textAug = textAug.replace('[', '')
            textAug = textAug.replace(']', '')
            df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
            df_temp = df_temp.append(df1, ignore_index = True)
            
    
    
    print("Augmenting Category 2 Text...")
    tot_cat2 = df_cat2.shape[0]
    for i in range(0,df_cat2.shape[0]):
        percent = ((i+1)/(tot_cat2)) * 100
        tot_percent = ((i+1+tot_cat1)/(total)) * 100
        print("Augmenting Category 2 Text " + str(i+1) + " / " + str(tot_cat2) + " (" + str(percent) + " %)" + " \t " + "Overall: " + str(tot_percent) + " %")
        
        text = df_cat2.loc[i,'text']

        out_Word2vec = t_Word2vec.augment(text)
        out_Wordnet = t_Wordnet.augment(text)

        out_Synonym = t_EDA.synonym_replacement(text)

        out_BERT = create_list_empty_strings(len_BERT)
        
        for k in range(0,len_BERT):
            out_BERT[k] = aug_bert.augment(text)

        
        name = df_cat1.loc[i, 'FileName'].replace(".txt","_aug.txt")
        timeframe = df_cat2.loc[i, 'TimeFrame']
        pos = df_cat2.loc[i, 'Positive score (+/-)']
        act = df_cat2.loc[i, 'Active score (+/-)']
        one_pred = df_cat2.loc[i, 'one pred']
        
        textAug = out_Word2vec.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        textAug = out_Wordnet.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        
        textAug = out_Synonym.replace("'", "")
        textAug = textAug.replace('"', '')
        textAug = textAug.replace('[', '')
        textAug = textAug.replace(']', '')
        df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
        df_temp = df_temp.append(df1, ignore_index = True)
        
        for k in range(0,len(out_BERT)):
            textAug = out_BERT[k][0].replace("'", "")
            textAug = textAug.replace('"', '')
            textAug = textAug.replace('[', '')
            textAug = textAug.replace(']', '')
            df1 = pd.DataFrame({"FileName":[name],"TimeFrame":[timeframe],"text":[textAug],"Positive score (+/-)":[pos],"Active score (+/-)":[act],"one pred":[one_pred]})
            df_temp = df_temp.append(df1, ignore_index = True)
            
    df = df.append(df_temp, ignore_index = True)
    df.to_csv(filename, index=None)

    
dataset_dir_path = 'Text_Dataset\\one pred'


nltk.download(['wordnet','punkt','averaged_perceptron_tagger'])

#!wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

model = gensim.models.KeyedVectors.load_word2vec_format('TextEmbedding/GoogleNews-vectors-negative300.bin.gz', binary=True)
t_Word2vec = Word2vec(model=model)
t_Wordnet = Wordnet()
t_Translate = Translate(src="en", to="fr")
t_EDA = EDA()

len_BERT = 7
TOPK=20 #default=100
ACT = 'insert' #"substitute"

test_sentence =  'Alright, were going to go ahead and assign tasks for everyone.'

aug_bert = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', 
    #device='cuda',
    action=ACT, top_k=TOPK)


augmentDataset(dataset_dir_path + '\\Text_Dataset_LATEST.csv')



