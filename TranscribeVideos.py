#import whisper
#import whisper_timestamped as whisper
#import whisper
import whisperx
from datetime import timedelta
import pandas as pd
import numpy as np
from numpy import nan 
import os

import pickle

import openai

import subprocess

from typing import Iterator, TextIO

def srt_format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def write_srt(transcript: Iterator[dict], file: TextIO):
    count = 0
    for segment in transcript:
        count +=1
        print(
            f"{count}\n"
            f"0{srt_format_timestamp(segment['start'])} --> {srt_format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->').strip()}\n",
            file=file,
            flush=True,
        )

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range") 



def transcribeVideo(folder, filename, idx, total_files, model, semester, direction):
    
    if direction == 'forward':
        print("Transcribing...")
        percent = round((idx/total_files) * 100, 2)
        
        print("Transcribing " + semester + " videos: "
            + str(idx + 1) + " / " + str(total_files)
            + " (" + str(percent) + " %) : "
            + filename.replace('.wav',''))
    
    elif direction == 'backward':
        print("Transcribing...")
        percent = round(((total_files - idx)/total_files) * 100, 2)
        
        print("Transcribing " + semester + " videos: "
            + str(total_files - idx) + " / " + str(total_files)
            + " (" + str(percent) + " %) : "
            + filename.replace('.wav',''))
    
    if not os.path.exists(folder + "\\" + filename.replace('.wav', '_segments.pkl')):
    
        #### For whisper ####
        #result = model.transcribe(folder + "\\" + filename, verbose=True, language="en")
        
        #### For whisperX (Tanzim edited) ####
        result = model.transcribe(folder + "\\" + filename, verbose=True, language="en")
        
        
        segments = result['segments']
        
        with open(folder + "\\" + filename.replace('.wav', '_segments.pkl'), 'wb') as f:  # open a text file
            pickle.dump(segments, f) # serialize the list
    else:
        with open(folder + "\\" + filename.replace('.wav', '_segments.pkl'), 'rb') as f:
            segments = pickle.load(f) # deserialize using load()
    
    
    with open(folder + "\\" + filename.replace('.wav', '_nonAligned.txt'), "w") as srt:
        write_srt(segments, file=srt)
        
        
    print("Initial transcription completed")
    
    print("Aligning....")
    
    ##### For alignment (WhisperX) #####
    
    if not os.path.exists(folder + "\\" + filename.replace('.wav', '_segments_aligned.pkl')):
    
        model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")
        result = whisperx.align(segments, model_a, metadata, folder + "\\" + filename, "cuda", return_char_alignments=False, verbose=True)

        #result = model.transcribe(folder + "\\" + filename, verbose=True, language="en")
        segments = result['segments']
        
        with open(folder + "\\" + filename.replace('.wav', '_segments_aligned.pkl'), 'wb') as f:  # open a text file
            pickle.dump(segments, f) # serialize the list
    else:
        with open(folder + "\\" + filename.replace('.wav', '_segments_aligned.pkl'), 'rb') as f:
            segments = pickle.load(f) # deserialize using load()

    # save SRT
    
    with open(folder + "\\" + filename.replace('.wav', '.txt'), "w") as srt:
        write_srt(segments, file=srt)




videos_folder = "videos_fall2022"

res_videos = []

for file in os.listdir(videos_folder):
    if file.endswith('.wav'):
        if not os.path.exists(videos_folder + "\\" + file.replace('.wav','.txt')):
            res_videos.append(file)


print("Found " + str(len(res_videos)) + " videos")
    
print("Loading model")

#### For whisperX (Tanzim edited) ####

model = whisperx.load_model("medium.en", device="cuda", compute_type="float16")

######################


#### For whisper ####

#model = whisper.load_model("medium.en")
#model = whisper.load_model("base.en")

######################

print("Model loaded")


for i in range(0, len(res_videos)):
    transcribeVideo(videos_folder, res_videos[i], i, len(res_videos), model, "Fall 2021", "forward")
  