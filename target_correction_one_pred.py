import os
import pandas as pd
from scipy.interpolate import interp1d
import math


def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list

def create_list_empty_ints(n):
    my_list = []
    for i in range(n):
        my_list.append(0)
    return my_list
    
def process_data_target(filename):
    print(filename)
    df = pd.read_csv(target_dir_path + '\\' + filename)
    time_df = df['Time']
    positive_df = df['Positive score (+/-)']
    active_df = df['Active score (+/-)']
    
    k = positive_df.values.tolist()
    
    pos_label = create_list_empty_ints(len(k))
    
    for i in range(0,len(k)):
        pos_label[i] = int(k[i])
        
    k = active_df.values.tolist()
    
    act_label = create_list_empty_ints(len(k))
    
    for i in range(0,len(k)):
        act_label[i] = int(k[i])
    
    fin_label = create_list_empty_ints(len(pos_label))
    
    for i in range(0,len(k)):
        if act_label[i] == 0 and pos_label[i] == 0:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 0 and pos_label[i] == 1:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 0 and pos_label[i] == 2:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 0 and pos_label[i] == 3:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 0 and pos_label[i] == 4:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 0:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 1 and pos_label[i] == 1:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 1 and pos_label[i] == 2:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 3:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 4:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 2 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 2 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 2 and pos_label[i] == 2:
            fin_label[i] = 0 #neutral
        elif act_label[i] == 2 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 2 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 3 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 3 and pos_label[i] == 2:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 4 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 4 and pos_label[i] == 2:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
 
    df['one pred'] = fin_label
    df.to_csv(target_dir_path + '\\' + filename, index=False)
    #print(df)

def process_data_dataset(filename):
    print(filename)
    df = pd.read_csv(dataset_dir_path + '\\' + filename)
    
    filename_df = df['FileName']
    timeframe_df = df['TimeFrame']
    text_df = df['text']
    positive_df = df['Positive score (+/-)']
    active_df = df['Active score (+/-)']
    
    k = positive_df.values.tolist()
    
    pos_label = create_list_empty_ints(len(k))
    
    for i in range(0,len(k)):
        pos_label[i] = int(k[i])
        
    k = active_df.values.tolist()
    
    act_label = create_list_empty_ints(len(k))
    
    for i in range(0,len(k)):
        act_label[i] = int(k[i])
    
    fin_label = create_list_empty_ints(len(pos_label))
    
    for i in range(0,len(k)):
        if act_label[i] == 0 and pos_label[i] == 0:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 0 and pos_label[i] == 1:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 0 and pos_label[i] == 2:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 0 and pos_label[i] == 3:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 0 and pos_label[i] == 4:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 0:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 1 and pos_label[i] == 1:
            fin_label[i] = 1 #neg-inact
        elif act_label[i] == 1 and pos_label[i] == 2:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 3:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 1 and pos_label[i] == 4:
            fin_label[i] = 2 #pos-inact
        elif act_label[i] == 2 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 2 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 2 and pos_label[i] == 2:
            fin_label[i] = 0 #neutral
        elif act_label[i] == 2 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 2 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 3 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 3 and pos_label[i] == 2:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 3 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 0:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 4 and pos_label[i] == 1:
            fin_label[i] = 4 #neg-act
        elif act_label[i] == 4 and pos_label[i] == 2:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 3:
            fin_label[i] = 3 #pos-act
        elif act_label[i] == 4 and pos_label[i] == 4:
            fin_label[i] = 3 #pos-act
 
    df['one pred'] = fin_label
    df.to_csv(dataset_dir_path + '\\' + filename, index=False)
    #print(df)
    
    
# Param range

prev_min_range = 0
prev_max_range = 10

min_range = 0
max_range = 4

# folder path
target_dir_path = 'videos'
#dataset_dir_path = 'Text_Dataset\\one pred'

# list to store files
target_res = []
dataset_res = []
# Iterate directory
for file in os.listdir(target_dir_path):
    # check only text files
    if file.endswith('.csv'):
        target_res.append(file)

# for file in os.listdir(dataset_dir_path):
    # # check only text files
    # if file.endswith('.csv'):
        # dataset_res.append(file)        



#print(dataset_res)

# for i in range(0,len(dataset_res)):
    # process_data_dataset(dataset_res[i])

for i in range(0,len(target_res)):
    process_data_target(target_res[i])
