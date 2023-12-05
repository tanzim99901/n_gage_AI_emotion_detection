from pydub import AudioSegment
import pandas as pd

from pydub.playback import play
import moviepy.editor as mp

import os

def prepAudioDataset(filename, idx, main_path):
    global df_main, total_files
    
    percent = round(((idx + 1)/total_files) * 100, 2)
    
    orig_filename = filename
    filename = filename.replace('.wav', '')
    
    label = int(filename[-1]) - 1
    
    print(str(idx) + " / " + str(total_files) + " (" + str(percent) + " %) : " + str(filename) + " \t " + "Label: " + str(label))
    
    emotion_labels = ['neg_deact', 'pos_deact', 'pos_act', 'neg_act']
    
    df_main.loc[idx,'emotion_label'] = emotion_labels[label]
    df_main.loc[idx,'emotion2'] = emotion_labels[label]
    df_main.loc[idx,'emotion3'] = emotion_labels[label]
    
    df_main.loc[idx,'source'] = 'PECAS'
    
    df_main.loc[idx,'actors'] = 'student'
    
    df_main.loc[idx,'path'] = os.path.split(os.path.abspath(orig_filename))[0] + "\\" + main_path + "\\" + orig_filename

main_path = "Audio Processing\\audioData"

column_names = ['emotion_label', 'source', 'actors', 'path', 'emotion2', 'emotion3']
df_main = pd.DataFrame(columns = column_names)

res = []

for file in os.listdir(main_path):

    if file.endswith('.wav'):
        res.append(file)

total_files = len(res)        
for i in range(0, len(res)):
    prepAudioDataset(res[i], i, main_path)

df_main.to_csv(main_path + '\\AudioDataset_excel.csv', index=None)
