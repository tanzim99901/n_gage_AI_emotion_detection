from pydub import AudioSegment
import pandas as pd

from pydub.playback import play
import moviepy.editor as mp

import os

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list
    
def createAudioDataset(filename):
    print(filename + "\n")
    audioFull = AudioSegment.from_file(directory + '\\' + filename, format="wav")
    
    textFilename = filename.replace('.wav','.txt')
    with open(directory + '\\' + textFilename) as f:
        textFile = f.readlines()
    #print(textFile)
    
    time_string = create_list_empty_strings(int(len(textFile)/4))
    text_string = create_list_empty_strings(int(len(textFile)/4))
    
    currText = ''
    counterText = 0
    
    for i in range(0,len(textFile)):
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

            time_string[counterText] = start_time + '>' + end_time
            
            text_string[counterText] = textFile[i+1]
            
            counterText = counterText + 1
    
    #print(text_string)
    #print(time_string)
    targetFilename = filename.replace('.wav','_target.csv')
    
    df_target = pd.read_csv(directory + '\\' + targetFilename)

    target_time_df = df_target['Time']
    target_df = df_target['one pred']
    #target_active_df = df_target['Active score (+/-)']

    target_time_string = target_time_df.values.tolist()
    target_string = target_df.values.tolist()
    #target_active_string = target_active_df.values.tolist()
    
    for i in range(0,len(target_time_string)):
        start = target_time_string[i][0] + target_time_string[i][1]
        end = target_time_string[i][3] + target_time_string[i][4]
        #print(start + '.........' + end)
        
        start_time = str(float(start) * 60 + 0.0001)
        end_time = str(float(end) * 60)
        
        target_time_string[i] = start_time + '>' + end_time
        #print(target_time_string[i])
    
    counter = 0
    for i in range(0,len(text_string)):
        match_found = False
        currText = text_string[i]
        currTime = time_string[i]
                    
        currText_transform = currText.replace('\n','')
        
        beginning = float(time_string[i][0:time_string[i].index('>')])
        #print(beginning)
        end = float(time_string[i][time_string[i].index('>')+1:])
        
        for j in range(0,len(target_time_string)):
            target_beginning = float(target_time_string[j][0:target_time_string[j].index('>')])
            target_end = float(target_time_string[j][target_time_string[j].index('>')+1:])
            
            if beginning >= target_beginning and end <= target_end:
                match_found = True
                label = target_string[j]
        
        if match_found == True:
            counter = counter + 1
            audioClip = audioFull[beginning*1000:end*1000]
            
            out_filename = filename.replace('.wav','_' + f"{i:03}" + '_' + str(label) + '.wav')
            
            print(out_filename)
            
            #audioClip.export(dataset_folder + '//' + str(label) + '//' + out_filename, format="wav")
            audioClip.export(dataset_folder + '//' + out_filename, format="wav")
            
            
        match_found = False

directory = 'videos'
parentAudioFolder = 'Audio processing'
childAudioFolder = 'audioData'
dataset_folder = parentAudioFolder + '//' + childAudioFolder

if not os.path.exists(parentAudioFolder):
    os.mkdir(parentAudioFolder)
    
if not os.path.exists(parentAudioFolder + '//' + childAudioFolder):
    os.mkdir(parentAudioFolder + '//' + childAudioFolder)
    
    
match_found = False

# list to store files
res = []
# Iterate directory
for file in os.listdir(directory):
    # check only text files
    if file.endswith('.wav'):
        res.append(file)
        
# # print(res)
for i in range(0,len(res)):
    createAudioDataset(res[i])


#createAudioDataset("Team2_teamsession2.wav")