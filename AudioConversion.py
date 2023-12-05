from pydub import AudioSegment

from pydub.playback import play
import moviepy.editor as mp

import os

def convertAudio(filename):
    print(filename + "\n")
    my_clip = mp.VideoFileClip(directory + '\\' + filename)
    
    out_filename = filename.replace('.mp4','.wav')
    
    my_clip.audio.write_audiofile(directory + '\\' + out_filename)


directory = 'videos'

# list to store files
res = []
# Iterate directory
for file in os.listdir(directory):
    # check only text files
    if file.endswith('.mp4'):
        res.append(file)
        
# print(res)
for i in range(0,len(res)):
    convertAudio(res[i])