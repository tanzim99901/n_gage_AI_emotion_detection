import pandas as pd
import numpy as np



def process_data(filename):
    print(filename)
    with open(dir_path + '\\' + filename) as f:
        textFile = f.readlines()
    time_string = create_list_empty_strings(int(len(textFile)/4))
    time_string_for_comp = create_list_empty_strings(int(len(textFile)/4))
    text_string = create_list_empty_strings(int(len(textFile)/4))
    filename_string = create_list_empty_strings(int(len(textFile)/4))
    positive_string = create_list_empty_strings(int(len(textFile)/4))
    active_string = create_list_empty_strings(int(len(textFile)/4))
    pred_string = create_list_empty_strings(int(len(textFile)/4))
    currText = ''
    counterText = 0
    
    targetFilename = filename.replace('.txt','_target.csv')
    
    df_target = pd.read_csv(dir_path + '\\' + targetFilename)

    target_time_df = df_target['Time']
    target_df = df_target['one pred']
    target_active_df = df_target['Active score (+/-)']
    target_positive_df = df_target['Positive score (+/-)']

    target_time_string = target_time_df.values.tolist()
    target_string = target_df.values.tolist()
    target_active_string = target_active_df.values.tolist()
    target_positive_string = target_positive_df.values.tolist()
    
    target_beginning = create_list_empty_strings(int(len(target_time_string)))
    target_end = create_list_empty_strings(int(len(target_time_string)))
    
    # for i in range(0,arr.shape[0]-1):
        # for j in range(0,arr.shape[1]-1):
            # arr[i,j] = ''
    
    for i in range(0,len(textFile)-1):
        if (i%4 == 1):
            hours = textFile[i][0] + textFile[i][1]

            minutes = textFile[i][3] + textFile[i][4]

            seconds = textFile[i][6] + textFile[i][7]

            seconds_dec = textFile[i][9] + textFile[i][10] + textFile[i][11]

            
            start_time = str(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) + '.' + seconds_dec

            
            
            hours = textFile[i][17] + textFile[i][18]

            minutes = textFile[i][20] + textFile[i][21]

            seconds = textFile[i][23] + textFile[i][24]

            seconds_dec = textFile[i][26] + textFile[i][27] + textFile[i][28]

            end_time = str(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) + '.' + seconds_dec
            
           
            time_string_for_comp[counterText] = start_time + '>' + end_time
            time_string[counterText] = textFile[i]

            text_string[counterText] = textFile[i+1]

            counterText = counterText + 1
    
    for i in range(0,len(target_time_string)):
        start = target_time_string[i][0] + target_time_string[i][1]
        end = target_time_string[i][3] + target_time_string[i][4]
        #print(start + '.........' + end)
        
        start_time = str(float(start) * 60 + 0.0001)
        end_time = str(float(end) * 60)
        
        target_time_string[i] = start_time + '>' + end_time
        target_beginning[i] = float(target_time_string[i][0:target_time_string[i].index('>')])
        target_end[i] = float(target_time_string[i][target_time_string[i].index('>')+1:])
        #print(target_time_string[i])
    
    counter = 0
    for i in range(0,len(text_string)):
        currText = text_string[i]
        currTime = time_string_for_comp[i]
                    
        currText_transform = currText.replace('\n','')
        
        beginning = float(time_string_for_comp[i][0:time_string_for_comp[i].index('>')])
        #print(beginning)
        end = float(time_string_for_comp[i][time_string_for_comp[i].index('>')+1:])
        
        for j in range(0,len(target_time_string)):
            #print(j)
            #print(len(target_time_string))
            #target_beginning = float(target_time_string[j][0:target_time_string[j].index('>')])
            #target_end = float(target_time_string[j][target_time_string[j].index('>')+1:])
            
            if beginning >= target_beginning[j] and end <= target_end[j]:
                match_found = True
                pred_string[i] = str(target_string[j])
                positive_string[i] = str(target_positive_string[j])
                active_string[i] = str(target_active_string[j])
            else:
                if j != len(target_time_string) - 1:
                    if beginning >= target_beginning[j] and end <= target_end[j+1]:
                        match_found = True
                        pred_string[i] = str(target_string[j])
                        positive_string[i] = str(target_positive_string[j])
                        active_string[i] = str(target_active_string[j])
        
        if match_found == True:
            counter = counter + 1
            
            
        match_found = False
        
    for i in range(0,len(time_string)):
        #filename_string[i] = filename.replace('videos\\','')
        filename_string[i] = filename
    
    
    #print(target_time_string[len(target_time_string)])
    all_data = pd.concat([pd.DataFrame(filename_string), pd.DataFrame(time_string), pd.DataFrame(text_string), pd.DataFrame(positive_string), pd.DataFrame(active_string), pd.DataFrame(pred_string)], axis=1, ignore_index=True)
    return all_data

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list
    
def create_list_empty_strings_multi(rows,columns):
    my_list = ['','','']
    for i in range(rows):
        my_list.append(my_list)
    return my_list
    
global Main_Data


import os

# folder path
dir_path = 'videos'

# list to store files
text_res = []
target_res = []
# Iterate directory
for file in os.listdir(dir_path):
    # check only text files
    if file.endswith('.txt'):
        text_res.append(file)

for file in os.listdir(dir_path):
    # check only text files
    if file.endswith('_target.csv'):
        target_res.append(file)
  


if not os.path.exists('Text_Dataset'):
    os.mkdir('Text_Dataset')
    
if not os.path.exists('Text_Dataset\\one pred'):
    os.mkdir('Text_Dataset\\one pred')

  
# print(len(text_res))

# for i in range(0,len(text_res)-1):
    # print(text_res[i])

data = [['', '', '', '', '', ''],['', '', '', '', '', '']]

#print(len(target_res))
#print(len(text_res))
 
# Create the pandas DataFrame
Main_Data = pd.DataFrame(data)
#print(Main_Data)

for i in range(0,len(text_res)):
    inter = process_data(text_res[i])
    Main_Data = pd.concat([Main_Data, inter], ignore_index=True)

Main_Data.columns =['FileName', 'TimeFrame', 'text', 'Positive score (+/-)', 'Active score (+/-)', 'one pred']
#print(Main_Data)
Main_Data = Main_Data.dropna()
Main_Data.to_csv('Text_dataset' + '\\' + 'Text_Dataset_LATEST.csv')
Main_Data.to_csv('Text_dataset' + '\\one pred\\' + 'Text_Dataset_LATEST.csv')






























# Main_Data = np.chararray(1,3)

# Team1_teamsession2 = 'videos\\Team1_teamsession2.txt'
# Team2_teamsession2
# Team3_teamsession1
# Team3_teamsession2
# Team4_teamsession1
# Team5_teamsession1
# Team7_teamsession2
# Team8_TeamSession1
# Team8_teamsession2
# Team9_teamsession1
# Team9_teamsession2
# Team10_teamsession1
# Team12_teamsession1
# Team12_teamsession2
# Team13_teamsession1
# Team13_teamsession2
# Team14_teamsession1
# Team15_teamsession1
# Team15_teamsession2_3s
# Team17_teamsession1
# Team18_teamsession1
# Team21_teamsession1
# Team21_teamsession2
# Team22_teamsession1
# Team22_teamsession2_3s
# Team23_teamsession1
# Team23_teamsession2
# Team24_teamsession1
# Team24_teamsession2
# Team25_teamsession2
# Team26_teamsession1
# Team26_teamsession2




# jack = process_data(Team1_teamsession2)

# john = pd.concat([jack,jack], ignore_index=True)
# print(john)
