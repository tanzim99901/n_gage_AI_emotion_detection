import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext

def process_data_target(filename):
    print(filename)
    df = pd.read_csv(target_dir_path + '\\' + filename)
    time_df = df['Time']
    positive_df = df['Positive score (+/-)']
    active_df = df['Active score (+/-)']
    
    m = interp1d([prev_min_range, prev_max_range], [min_range, max_range])
    
    for i in range(0,positive_df.shape[0]):
        to_float = float(positive_df.iloc[i])
            
        transformed = math.floor(to_float)
        
        positive_df.iloc[i] = str(transformed)
        
        
    for i in range(0,active_df.shape[0]):
        to_float = float(active_df.iloc[i])
            
        transformed = math.floor(to_float)
        
        active_df.iloc[i] = str(transformed)
        

    
    Main_Data = pd.concat([time_df, positive_df, active_df], axis = 1)
    Main_Data.to_csv(target_dir_path + '\\' + filename, index=False)

def process_data_dataset(filename):
    print(filename)
    out_filename_train = filename.replace('.csv','_train.txt')
    out_filename_test = filename.replace('.csv','_test.txt')
    df = pd.read_csv(dataset_dir_path + '\\' + filename)
    
    train_df = df.sample(frac = 0.8)
    train_df.to_csv(dataset_dir_path + '\\' + out_filename_train.replace('.txt','.csv'), index=None)
    test_df = df.drop(train_df.index)
    #test_df = df
    test_df.to_csv(dataset_dir_path + '\\' + out_filename_test.replace('.txt','.csv'), index=None)
    
    # print(test)
    
    # print(train)
    
    
    #### create training dataset ####
    filename_df = train_df['FileName']
    timeframe_df = train_df['TimeFrame']
    text_df = train_df['text']
    label_df = train_df['one pred']
    
    for i in range(0,filename_df.shape[0]):
        inter = str(text_df.iloc[i].replace('"',''))
        inter = inter.replace('\n','')
        inter = inter + ' ' + '__label__' + str(int(label_df.iloc[i]))
        text_df.iloc[i] = inter

        #print(text_df.iloc[i])

    text_df.to_csv(dataset_dir_path + '\\' + out_filename_train, header=None, index=None, sep='\n', mode='a')
    
    
    
    #### create test dataset ####
    filename_df = test_df['FileName']
    timeframe_df = test_df['TimeFrame']
    text_df = test_df['text']
    label_df = test_df['one pred']
    
    for i in range(0,filename_df.shape[0]):
        inter = str(text_df.iloc[i].replace('"',''))
        inter = inter.replace('\n','')
        inter = inter + ' ' + '__label__' + str(int(label_df.iloc[i]))
        text_df.iloc[i] = inter

        #print(text_df.iloc[i])

    text_df.to_csv(dataset_dir_path + '\\' + out_filename_test, header=None, index=None, sep='\n', mode='a')
    
    
# Param range

prev_min_range = 0
prev_max_range = 10

min_range = 0
max_range = 4

# folder path
target_dir_path = 'videos'
dataset_dir_path = 'Text_Dataset\\one pred'

# list to store files
target_res = []
dataset_res = []
# Iterate directory
for file in os.listdir(target_dir_path):
    # check only text files
    if file.endswith('.csv'):
        target_res.append(file)

for file in os.listdir(dataset_dir_path):
    # check only text files
    if file.endswith('_LATEST.csv'):
        dataset_res.append(file)        



#print(dataset_res)

for i in range(0,len(dataset_res)):
    process_data_dataset(dataset_res[i])

# for i in range(0,len(target_res)):
    # process_data_target(target_res[i])
