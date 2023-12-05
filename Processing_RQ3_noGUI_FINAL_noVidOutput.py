import sys
import random
import time
import cv2

import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms, utils

from ffpyplayer.player import MediaPlayer

import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


import math

import pickle

import pandas as pd

import fasttext

from pydub import AudioSegment

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wave

import librosa
from python_speech_features import *

#import loupe_keras_tanzim as lpk

import tensorflow.compat.v1 as tf

from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import random
import itertools

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam

import pickle

from moviepy.editor import *

import subprocess


def stop():
    global ThreadActive
    ThreadActive = False
    print("Stop video")
    return False
    
def close(cap):
    global ThreadActive
    ThreadActive = False
    print("Close video")
    cap.release()
    cv2.destroyAllWindows()
    #self.quit()
    
def detect_face(video_capture, audio_capture):
    # Grab a single frame of video
    ret, frame = video_capture.read()
    audio_frame, val = audio_capture.get_frame()
    if audio_frame is not None:
        a_a, audioCurrDur = audio_frame
    #print(audioCurrDur)
    # if frame is not None:
        # b_b, videoCurrDur = frame
    # print(videoCurrDur)
    h, w, ret = frame.shape
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_loc = face_recognition.face_locations(rgb_small_frame)
    if len(face_loc) == 0:
        return 0,0,0,0,frame,audio_frame,False
    else:
        tanzim = 1
        #print(face_loc)
        #face_loc = face_loc[0]
    
    top = np.zeros(len(face_loc))
    bottom = np.zeros(len(face_loc))
    left = np.zeros(len(face_loc))
    right = np.zeros(len(face_loc))
    #crop_frame = np.zeros((len(face_loc),3))
    
    for idx, x in enumerate(face_loc):
        top[idx], right[idx], bottom[idx], left[idx] = x
        h_len = bottom[idx] - top[idx]
        w_len = right[idx] - left[idx]
        new_len = max(h_len, w_len) * 1.5
        center = np.array([top[idx] + h_len / 2, left[idx] + w_len / 2])
        scale = 4
        top[idx] = (int)(center[0] - new_len / 2) * scale
        bottom[idx] = (int)(center[0] + new_len / 2) * scale
        left[idx] = (int)(center[1] - new_len / 2) * scale
        right[idx] = (int)(center[1] + new_len / 2) * scale
        top[idx] = int(max(0, top[idx]))
        bottom[idx] = int(min(h, bottom[idx]))
        left[idx] = int(max(0, left[idx]))
        right[idx] = int(min(w, right[idx]))
    return top,bottom,left,right,frame,audio_frame,ret
    
def write_video(filename, all_data_out, fps, img_size, img_array):
    time.sleep(1)
    
    out_filename = filename.replace('.mp4','') + '_processed.mp4'
    a_out_filename = filename.replace('.mp4','') + '_framed.mp4'
    
    print(out_filename)


    all_data_out.to_csv(out_filename.replace('.mp4','') + "_all_data.csv", index=False)
    
    # out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc(*'XVID'),fps, img_size)

    # for i in range(len(img_array)):
        # out.write(img_array[i])
    # out.release()
    
    
    
    processFin = True
    time.sleep(10)
    
   
    
    
    
    
    
    
def MixAudioVideo(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    fps = video.fps
    #print(fps)
    video = video.set_audio(AudioFileClip(audio_path))
    video.write_videofile(output_path)
    
    #del video, fps

# Function to find division without 
# using '/' operator
def division(num1, num2):
     
    if (num1 == 0): return 0
    if (num2 == 0): return INT_MAX
     
    negResult = 0
     
    # Handling negative numbers
    if (num1 < 0):
        num1 = - num1
         
        if (num2 < 0):
            num2 = - num2
        else:
            negResult = True
                 
    elif (num2 < 0):
        num2 = - num2
        negResult = True
     
    # if num1 is greater than equal to num2
    # subtract num2 from num1 and increase
    # quotient by one.
    quotient = 0
 
    while (num1 >= num2):
        num1 = num1 - num2
        quotient += 1
     
    # checking if neg equals to 1 then
    # making quotient negative
    if (negResult):
        quotient = - quotient
    return quotient
    
        
def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list

def get_audio_data(path, calculate_db=False, calculate_mfccs=False, plots=False):
    data, sampling_rate = librosa.load(path, sr=44100)
    Xdb = None
    if calculate_db:
        X = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(X))
    mfccs = None
    if calculate_mfccs:
        #mfccs = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc = 40)
        
        mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc = 40)
        
        #librosa.feature.melspectrogram(y=data, n_mfcc = 40,sr=sr)
    
    if calculate_db and plots:
        fig, ax = plt.subplots(1,2,figsize=(16, 3))
        plt.subplot(121)
        #librosa.display.waveplot(data, sr=sampling_rate)
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.subplot(122)
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
        plt.show()
    elif plots:
        #librosa.display.waveplot(data, sr=sampling_rate)
        librosa.display.waveshow(data, sr=sampling_rate)

    return (data, Xdb, mfccs)
 
def predictAudioClass(file_loc, model):
    NN_data = []
    a1, a2, a3 = get_audio_data(file_loc, calculate_db=True)
    img = np.stack((a2,) * 3,-1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    NN_data.append(grayImage)

    mel_images = np.array(NN_data)
    rgb_batch = np.repeat(mel_images[..., np.newaxis], 3, -1)
    female_idxs = 0
    mel_images_female = rgb_batch[female_idxs,:,:,:]
    mel_images_female = np.reshape(mel_images_female, (1, mel_images_female.shape[0], mel_images_female.shape[1], mel_images_female.shape[2]))
    y_pred = model.predict(mel_images_female, verbose=0).argmax(axis=1)
    
    # 0 = neg_act -> 4
    # 1 = neg_deact -> 1
    # 2 = pos_act -> 3
    # 3 = pos_deact -> 2
    if y_pred[0] == 0:
        audio_pred = 4
    elif y_pred[0] == 1:
        audio_pred = 1
    elif y_pred[0] == 2:
        audio_pred = 3
    elif y_pred[0] == 3:
        audio_pred = 2
    else:
        audio_pred = 0
        
    return audio_pred

def clearVars():
    
    # declaring multiple variables
    a, b, c = 5, 10, 15
    print("variables: a: ", a, " b: ", b, " c: ", c)

    #inititalizing d with dir()
    d = dir()
    #printing the directory
    print(d)
    

            
def processFile(filename,n_file,total_files):
    global cluster_size
    global audio_model, text_model, video_model
    global w_vid, w_aud, w_text
    global s_negAct, s_negInact, s_posAct, s_posInact


    
    
    
    #### Scoring variables ####
    
    w_vid = 0.33
    w_text = 0.33
    w_aud = 0.33

    s_posAct = 1
    s_posInact = 0.5
    s_negInact = -0.5
    s_negAct = -1 
    
    ##### Load models start #####
    pretrained_model = tf.keras.applications.DenseNet201(include_top=False, 
                                                         weights='imagenet', 
                                                         input_shape=(224,224,3))
    for layer in pretrained_model.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    audio_model = tf.keras.models.Sequential()
    audio_model.add(pretrained_model)
    audio_model.add(tf.keras.layers.GlobalAveragePooling2D())
    audio_model.add(tf.keras.layers.Flatten())
    audio_model.add(tf.keras.layers.Dense(256))
    audio_model.add(tf.keras.layers.Dropout(0.2))
    audio_model.add(tf.keras.layers.Dense(128))
    audio_model.add(tf.keras.layers.Dropout(0.1))
    audio_model.add(tf.keras.layers.Dense(4, activation='softmax'))
    audio_model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    audio_model_weights_filename = 'Audio_Model_CNN.hdf5'
    audio_model.load_weights(audio_model_weights_filename)
    
    
    
    video_model_filename = 'test_model_alpha_0'
    
    text_model_filename = "fasttext_one_pred_model.bin"
    text_model = fasttext.load_model(text_model_filename)
    
    #video_model = torch.load(video_model_filename, map_location=torch.device('cpu'))
    video_model = torch.load(video_model_filename, map_location=torch.device('cuda'))
    video_model.eval()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    
    ##### Load models end #####








    
    
    
    
    
    
    
    
    img_array = []
    framed_img_array = []
    textOn = False
    fileFound = False
    cluster_size = 16
    Init_Frame = 0
    vidFrames = 0
    vidLength = 0
    audioCurrDur = 0
    audioStartDur = 0
    videoCurrDur = 0
    videoStartDur = 0
    processFin = False

    ######## Load transcript file 
    
    textFilename = filename.replace('.mp4','.txt')
    with open(textFilename) as f:
        textFile = f.readlines()
    #print(len(textFile))
    time_string = create_list_empty_strings(int(len(textFile)/4))
    text_string = create_list_empty_strings(int(len(textFile)/4))
    
    currText = ''
    counterText = 0
    
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
            
           
            time_string[counterText] = start_time + '>' + end_time
            
            text_string[counterText] = textFile[i+1]
            counterText = counterText + 1
    ##############





    # ###### Load target file #####
    
    # targetFilename = filename.replace('.mp4','_target.csv')

    # df_target = pd.read_csv(targetFilename)

    # target_time_df = df_target['Time']
    # target_df = df_target['one pred']
    # target_active_df = df_target['Active score (+/-)']

    # target_time_string = target_time_df.values.tolist()
    # target_string = target_df.values.tolist()
    # target_active_string = target_active_df.values.tolist()


    # for i in range(0,len(target_time_string)):
        # start = target_time_string[i][0] + target_time_string[i][1]
        # end = target_time_string[i][3] + target_time_string[i][4]
        # #print(start + '.........' + end)
        
        # start_time = str(float(start) * 60 + 0.0001)
        # end_time = str(float(end) * 60)
        
        # target_time_string[i] = start_time + '>' + end_time
        # #print(target_time_string[i])
    
    # ################################
    
    
    
    
    
    
    textOn = False
    found_text = False
    found_text_2 = False
    found_text_0 = False
    found_text_1 = False
    found_text_3 = False
    found_text_4 = False
    found_text_5 = False
    found_text_6 = False
    found_text_7 = False
    found_text_8 = False
    found_text_9 = False
    found_text_10 = False
    
    ###### Load Audio File #######
    
    audioFilename = filename.replace('.mp4','.wav')
    audioFile = AudioSegment.from_file(audioFilename, format="wav")
    
    audioDirectoryname = audioFilename.replace('.wav','_audioClips')
    
    if not os.path.exists(audioDirectoryname):
        os.mkdir(audioDirectoryname)
    
    print("Parsing audio...")
    
    counter = 0
    for i in range(0,len(text_string)):
        currText = text_string[i]
        currTime = time_string[i]
                    
        currText_transform = currText.replace('\n','')
        
        beginning = float(time_string[i][0:time_string[i].index('>')])
        #print(beginning)
        end = float(time_string[i][time_string[i].index('>')+1:])
        audioClip = audioFile[beginning*1000:end*1000]
            
        out_filename = audioFilename.replace('.wav','_audioClips\\' + str(i) + '.wav')
        
        #print(out_filename)
        #print(i)
        
        audioClip.export(out_filename, format="wav")
    
    
    time.sleep(4)
    
    ###### Load Audio File ends #######
    
    
    
    
    text_positive_pred = 0.0
    text_active_pred = 0.0

    origin_x = 2300
    origin_y = 350
    
    positive_end_x = 2600
    positive_end_y = 350
    
    active_end_x = 2300
    active_end_y = 50
    
    negative_end_x = 2000
    negative_end_y = 350
    
    passive_end_x = 2300
    passive_end_y = 650
    
    min_param = 0
    max_param = 10
    
    video_point_color = (255, 0, 0)
    audio_point_color = (0, 150, 150)
    text_point_color = (0, 255, 0)
    target_box_color = (0, 0, 190)
    
    dead_x = origin_x
    dead_y = origin_y
    
    max_dist = math.dist((positive_end_x, active_end_y), (negative_end_x, passive_end_y))
    
    ThreadActive = True
    
    Capture = cv2.VideoCapture(filename)
    time.sleep(2)
    audioSource = MediaPlayer(filename)
    fps = Capture.get(cv2.CAP_PROP_FPS)
    sleep_ms = int(np.round((1/fps)*1000))
    vidFrames = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))
    vidLength = vidFrames/fps
    print("Video FPS: " + str(fps))
    print("Video Frames: " + str(vidFrames))
    print("Video Duration (s): " + str(vidLength))
    
    column_names = ['Frame', 'Distance from target', 'Loc_x', "Loc_y", "Positive score", "Active score", "Error percent"]
    video_data_out_np = np.zeros((vidFrames,len(column_names)))
    text_data_out_np = np.zeros((vidFrames,len(column_names)))
    text_data_out = pd.DataFrame(data = text_data_out_np, columns = column_names)
    video_data_out = pd.DataFrame(data = video_data_out_np, columns = column_names)
    
    final_column_names = ['Frame', 'Has face?',
    'Vid_x', 'Vid_y', 'Video class',
    'Text_x_0', 'Text_y_0', 'Text_x_1', 'Text_y_1', 'Text_x', 'Text_y', 
    'Text_x_3', 'Text_y_3', 'Text_x_4', 'Text_y_4', 'Text_x_5', 'Text_y_5', 
    'Text_x_6', 'Text_y_6', 'Text_x_7', 'Text_y_7', 'Text_x_8', 'Text_y_8', 
    'Text_x_9', 'Text_y_9', 'Text_x_10', 'Text_y_10',  
    'Aud_x_0', 'Aud_y_0', 'Aud_x_1', 'Aud_y_1', 'Aud_x', 'Aud_y', 'Aud_x_3', 'Aud_y_3',
    'Aud_x_4', 'Aud_y_4', 'Aud_x_5', 'Aud_y_5', 'Aud_x_6', 'Aud_y_6',
    'Aud_x_7', 'Aud_y_7', 'Aud_x_8', 'Aud_y_8', 'Aud_x_9', 'Aud_y_9',
    'Aud_x_10', 'Aud_y_10',  
    'Text class_0', 'Text class_1', 'Text class', 'Text class_3', 'Text class_4',
    'Text class_5', 'Text class_6', 'Text class_7', 'Text class_8', 'Text class_9',
    'Text class_10', 
    'Audio class_0', 'Audio class_1', 'Audio class', 'Audio class_3', 'Audio class_4',
    'Audio class_5', 'Audio class_6', 'Audio class_7', 'Audio class_8', 'Audio class_9',
    'Audio class_10', 
    'Mismatch_0', 'Mismatch_1', 'Mismatch_2', 'Mismatch_3', 'Mismatch_4',
    'Mismatch_5', 'Mismatch_6', 'Mismatch_7', 'Mismatch_8', 'Mismatch_9',
    'Mismatch_10',
    ]
    all_data_out_np = np.zeros((vidFrames,len(final_column_names)))
    all_data_out = pd.DataFrame(data = all_data_out_np, columns = final_column_names)
          
          
    time.sleep(2)
    audioLength = float(audioSource.get_metadata()['duration'])
    fps_num = int(audioSource.get_metadata()['frame_rate'][0]) * 1
    fps_den = int(audioSource.get_metadata()['frame_rate'][1]) * 1
    audioFps = division(fps_num, fps_den)
    audioFrames = int(audioLength * audioFps)
    print("Audio FPS: " + str(audioFps))
    print("Audio Frames: " + str(audioFrames))
    print("Audio Duration (s): " + str(audioLength))
    audioStartDur = audioSource.get_pts()
    
    emotions = ['Neg-deact', 'Pos-deact', 'Pos-act', 'Neg-acti']
    
    textOn = False
    textProcessed = False
    text_pred = 0
    audio_pred = 0
    
    prev_text_pred = 0
    prev_audio_pred = 0
    
    video_pred = 0
    
    text_pred_0 = 0
    text_pred_1 = 0
    text_pred_2 = 0
    text_pred_3 = 0
    text_pred_4 = 0
    text_pred_5 = 0
    text_pred_6 = 0
    text_pred_7 = 0
    text_pred_8 = 0
    text_pred_9 = 0
    text_pred_10 = 0
    
    audio_pred_0 = 0
    audio_pred_1 = 0
    audio_pred_2 = 0
    audio_pred_3 = 0
    audio_pred_4 = 0
    audio_pred_5 = 0
    audio_pred_6 = 0
    audio_pred_7 = 0
    audio_pred_8 = 0
    audio_pred_9 = 0
    audio_pred_10 = 0
    
    while ThreadActive:
        if int(Capture.get(cv2.CAP_PROP_POS_FRAMES)) >= vidFrames:
            ThreadActive = stop()
            close(Capture)
            write_video(filename, all_data_out, fps, img_size, img_array)
            break
        
        t,b,l,r,w_frame,a_frame,ret = detect_face(Capture, audioSource)
        
        #main_frame = w_frame
        
        ### Plot graph for video
        
        top_y = 50
        bottom_y = 650
        left_x = 2000
        right_x = 2600
        half_length = 300
        
        w_frame = cv2.copyMakeBorder(w_frame, 0, 0, 0, 700, cv2.BORDER_CONSTANT, None, value = [255,255,255])
        
        w_frame = cv2.line(w_frame, (left_x + half_length, top_y), (right_x - half_length, bottom_y), (0,0,0), 9)  # The Y-axis (length 600)
        w_frame = cv2.line(w_frame, (left_x, top_y + half_length), (right_x, bottom_y - half_length), (0,0,0), 9) # The X-axis (length 600)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        ################
        
        ### Get audio and video current duration
        videoCurrDur = Capture.get(cv2.CAP_PROP_POS_FRAMES) / Capture.get(cv2.CAP_PROP_FPS)

        ######## TEXT AND AUDIO PROCESSING HERE ########
        
        found_text_0 = False
        found_text_1 = False
        found_text_2 = False
        found_text_3 = False
        found_text_4 = False
        found_text_5 = False
        found_text_6 = False
        found_text_7 = False
        found_text_8 = False
        found_text_9 = False
        found_text_10 = False

        for j in range(0,len(time_string)):
            beginning = float(time_string[j][0:time_string[j].index('>')])
            end = float(time_string[j][time_string[j].index('>')+1:])
            
            if(videoCurrDur >= beginning and videoCurrDur <= end):
                found_text_0 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                #print("For 0 s validity: " + str(currText_transform))
                jack = text_model.predict(currText_transform)
                tom = jack[0][0].replace('__label__','')
                
                ######
                text_pred_0 = int(tom)
                prev_text_pred = text_pred_0
                ######
                
                
                #######
                audio_pred_0 = predictAudioClass(audioDirectoryname + '//' + str(j) + '.wav', audio_model)
                prev_audio_pred = audio_pred_0
                #######
      
            if(videoCurrDur >= beginning and videoCurrDur <= end + 2.00):
                found_text_2 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_2 = prev_text_pred
                audio_pred_2 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 1.00):
                found_text_1 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_1 = prev_text_pred
                audio_pred_1 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 3.00):
                found_text_3 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_3 = prev_text_pred
                audio_pred_3 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 4.00):
                found_text_4 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_4 = prev_text_pred
                audio_pred_4 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 5.00):
                found_text_5 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_5 = prev_text_pred
                audio_pred_5 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 6.00):
                found_text_6 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_6 = prev_text_pred
                audio_pred_6 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 7.00):
                found_text_7 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_7 = prev_text_pred
                audio_pred_7 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 8.00):
                found_text_8 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_8 = prev_text_pred
                audio_pred_8 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 9.00):
                found_text_9 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_9 = prev_text_pred
                audio_pred_9 = prev_audio_pred
                    
            if(videoCurrDur >= beginning and videoCurrDur <= end + 10.00):
                found_text_10 = True
                currText = text_string[j]
                currText_transform = currText.replace('\n','')
                
                text_pred_10 = prev_text_pred
                audio_pred_10 = prev_audio_pred
              
    
        if found_text_0 == False:
            text_pred_0 = 0
            audio_pred_0  = 0
            
        if found_text_1 == False:
            text_pred_1 = 0
            audio_pred_1  = 0
            
        if found_text_2 == False:
            text_pred_2 = 0
            audio_pred_2  = 0
            
        if found_text_3 == False:
            text_pred_3 = 0
            audio_pred_3  = 0
            
        if found_text_4 == False:
            text_pred_4 = 0
            audio_pred_4  = 0
            
        if found_text_5 == False:
            text_pred_5 = 0
            audio_pred_5  = 0
            
        if found_text_6 == False:
            text_pred_6 = 0
            audio_pred_6  = 0
            
        if found_text_7 == False:
            text_pred_7 = 0
            audio_pred_7  = 0
            
        if found_text_8 == False:
            text_pred_8 = 0
            audio_pred_8  = 0
            
        if found_text_9 == False:
            text_pred_9 = 0
            audio_pred_9  = 0
            
        if found_text_10 == False:
            text_pred_10 = 0
            audio_pred_10  = 0
        
        
        ######## TEXT AND AUDIO PROCESSING END ########
        
      
        if type(t) is not int:
            preds_array = np.zeros((t.shape[0],), dtype=int)
            scores_array = np.zeros(t.shape[0])
            for idx, x in enumerate(t):
                c_frame = w_frame[int(t[idx]):int(b[idx]), int(l[idx]):int(r[idx])]
                #a_frame = main_frame[int(t[idx]):int(b[idx]), int(l[idx]):int(r[idx])]
                if c_frame is not None:

                    img = data_transforms(c_frame)
                    img = img.cuda()
                    scores = video_model(img.unsqueeze(0))
                    #print(scores)
                    _, preds = scores.max(1)
                    #print(_)
                    preds_array[idx] = int(preds.item())
                    scores_array[idx] = abs(_.item())

                        
                    if preds_array[idx] == 0: # Negative-deactivated
                        color = (0, 0, 255)
                        textColor = (255,255,255)
                    elif preds_array[idx] == 1: # Positive-deactivated
                        color = (0, 255, 0)
                        textColor = (0,0,0)
                    elif preds_array[idx] == 2: # Positive-activated
                        color = (255, 0, 0)
                        textColor = (255,255,255)
                    elif preds_array[idx] == 3: # Negative-activated
                        color = (255, 133, 233)
                        textColor = (0,0,0)
                    else:
                        color = (0, 0, 0)
                        textColor = (255,255,255)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.rectangle(w_frame, (int(l[idx]), int(t[idx])), (int(r[idx]), int(b[idx])), color, 2)
                    cv2.rectangle(w_frame, (int(l[idx]), int(b[idx]) - 35), (int(r[idx]), int(b[idx])), color, cv2.FILLED)
                    cv2.putText(w_frame, emotions[preds], (int(l[idx]) + 6, int(b[idx]) - 6), font, 1.0, textColor, 2)
                    
                    # cv2.rectangle(main_frame, (int(l[idx]), int(t[idx])), (int(r[idx]), int(b[idx])), color, 2)
                    # cv2.rectangle(main_frame, (int(l[idx]), int(b[idx]) - 35), (int(r[idx]), int(b[idx])), color, cv2.FILLED)
                    # cv2.putText(main_frame, emotions[preds], (int(l[idx]) + 6, int(b[idx]) - 6), font, 1.0, textColor, 2)
                    
            ##### MULTI USER VIDEO METHOD START #####
            # # 1 = pos act, 2 = neg-act, 3 = pos inact, 4 = neg-inact
            
            video_x_arr = np.zeros(len(preds_array),dtype=int)
            video_y_arr = np.zeros(len(preds_array),dtype=int)
            
            for i in range(0,len(video_x_arr)):
                
                if preds_array[i] == 0: # Negative-deactivated
                    video_x_arr[i] = int(origin_x - ((origin_x - negative_end_x)/2))
                    video_y_arr[i] = int(passive_end_y - (passive_end_y - origin_y)/2)
                elif preds_array[i] == 1: # Positive-deactivated
                    video_x_arr[i] = int(positive_end_x - ((positive_end_x - origin_x)/2))
                    video_y_arr[i] = int(passive_end_y - (passive_end_y - origin_y)/2)
                elif preds_array[i] == 2: # Positive-activated
                    video_x_arr[i] = int(positive_end_x - ((positive_end_x - origin_x)/2))
                    video_y_arr[i] = int(origin_y - ((origin_y - active_end_y)/2))
                elif preds_array[i] == 3: # Negative-activated
                    video_x_arr[i] = int(origin_x - ((origin_x - negative_end_x)/2))
                    video_y_arr[i] = int(origin_y - ((origin_y - active_end_y)/2))
                else:
                    video_x_arr[i] = dead_x
                    video_y_arr[i] = dead_y
                    
                if video_x_arr[i] == 0:
                    video_x_arr[i] = dead_x
                if video_y_arr[i] == 0:
                    video_y_arr[i] = dead_y
                    
            video_x = int(round(np.average(video_x_arr)))
            video_y = int(round(np.average(video_y_arr)))
            
            if video_x == 0:
                video_x = dead_x
            if video_y == 0:
                video_y = dead_y
            
            
            
            counter_1 = 0
            counter_2 = 0
            counter_3 = 0
            counter_4 = 0
            
            for i in range(0,len(preds_array)):
                
                if preds_array[i] == 0:
                    counter_1 += 1
                
                if preds_array[i] == 1:
                    counter_2 += 1
                
                if preds_array[i] == 2:
                    counter_3 += 1
                
                if preds_array[i] == 3:
                    counter_4 += 1
                    
            
            if len(preds_array) == 1:
                video_pred = preds_array[0] + 1
                
            else:
                if max([counter_1, counter_2, counter_3, counter_4]) == 1:
                    video_pred = 0
                
                else:
                    max_idx = [counter_1, counter_2, counter_3, counter_4].index(max([counter_1, counter_2, counter_3, counter_4]))
                    
                    if max_idx == 0:
                        video_pred = 1
                    elif max_idx == 1:
                        video_pred = 2
                    elif max_idx == 2:
                        video_pred = 3
                    elif max_idx == 3:
                        video_pred = 4

            
            hasFace = 1
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Frame'] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Has face?'] = str(hasFace)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_x'] = str(video_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_y'] = str(video_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video class'] = str(video_pred)

        
        else:
            video_x = dead_x
            video_y = dead_y
            
            video_pred = 0
            video_correct = 0
            hasFace = 0
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Frame'] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Has face?'] = str(hasFace)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_x'] = str(video_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_y'] = str(video_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video class'] = str(video_pred)

        ############# Draw video point
        
        if type(video_x) is not int:
            for i in range(0,len(video_x)):
                if video_x[i] != dead_x:
                    w_frame = cv2.circle(w_frame, (video_x[i], video_y[i]), 20, video_point_color, -1)
        else:
            if video_x != dead_x:
                w_frame = cv2.circle(w_frame, (video_x, video_y), 20, video_point_color, -1)
                
        
        
        ####### Put text prediction visualization here #######
        
        text_offset = 40
        audio_offset = 40
        
        #### For 2 second validity
        
        if text_pred_2 == 0: # neutral
            text_x = dead_x
            text_y = dead_y
        elif text_pred_2 == 1: # neg inact
            text_x = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_2 == 2: # pos inact 
            text_x = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_2 == 3: # pos act
            text_x = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_2 == 4: # neg act
            text_x = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        

            
        if text_x != dead_x:
            w_frame = cv2.circle(w_frame, (text_x, text_y), 20, text_point_color, -1)

        # if text_pred_2 == target_label:
            # text_correct = 1
        # else:
            # text_correct = 0
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x'] = str(text_x)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y'] = str(text_y)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class'] = str(text_pred_2)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct?'] = str(text_correct)


        #### For 0 second validity
        
        if text_pred_0 == 0: # neutral
            text_x_0 = dead_x
            text_y_0 = dead_y
        elif text_pred_0 == 1: # neg inact
            text_x_0 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_0 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_0 == 2: # pos inact 
            text_x_0 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_0 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_0 == 3: # pos act
            text_x_0 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_0 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_0 == 4: # neg act
            text_x_0 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_0 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_0 == target_label:
            # text_correct_0 = 1
        # else:
            # text_correct_0 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_0'] = str(text_x_0)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_0'] = str(text_y_0)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_0'] = str(text_pred_0)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_0?'] = str(text_correct_0)
        
        
        #### For 1 second validity
        
        if text_pred_1 == 0: # neutral
            text_x_1 = dead_x
            text_y_1 = dead_y
        elif text_pred_1 == 1: # neg inact
            text_x_1 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_1 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_1 == 2: # pos inact 
            text_x_1 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_1 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_1 == 3: # pos act
            text_x_1 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_1 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_1 == 4: # neg act
            text_x_1 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_1 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_1 == target_label:
            # text_correct_1 = 1
        # else:
            # text_correct_1 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_1'] = str(text_x_1)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_1'] = str(text_y_1)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_1'] = str(text_pred_1)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_1?'] = str(text_correct_1)

        
        #### For 3 second validity
        
        if text_pred_3 == 0: # neutral
            text_x_3 = dead_x
            text_y_3 = dead_y
        elif text_pred_3 == 1: # neg inact
            text_x_3 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_3 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_3 == 2: # pos inact 
            text_x_3 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_3 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_3 == 3: # pos act
            text_x_3 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_3 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_3 == 4: # neg act
            text_x_3 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_3 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_3 == target_label:
            # text_correct_3 = 1
        # else:
            # text_correct_3 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_3'] = str(text_x_3)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_3'] = str(text_y_3)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_3'] = str(text_pred_3)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_3?'] = str(text_correct_3)
        
        
        #### For 4 second validity
        
        if text_pred_4 == 0: # neutral
            text_x_4 = dead_x
            text_y_4 = dead_y
        elif text_pred_4 == 1: # neg inact
            text_x_4 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_4 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_4 == 2: # pos inact 
            text_x_4 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_4 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_4 == 3: # pos act
            text_x_4 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_4 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_4 == 4: # neg act
            text_x_4 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_4 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_4 == target_label:
            # text_correct_4 = 1
        # else:
            # text_correct_4 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_4'] = str(text_x_4)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_4'] = str(text_y_4)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_4'] = str(text_pred_4)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_4?'] = str(text_correct_4)
        
        
        #### For 5 second validity
        
        if text_pred_5 == 0: # neutral
            text_x_5 = dead_x
            text_y_5 = dead_y
        elif text_pred_5 == 1: # neg inact
            text_x_5 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_5 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_5 == 2: # pos inact 
            text_x_5 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_5 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_5 == 3: # pos act
            text_x_5 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_5 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_5 == 4: # neg act
            text_x_5 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_5 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_5 == target_label:
            # text_correct_5 = 1
        # else:
            # text_correct_5 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_5'] = str(text_x_5)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_5'] = str(text_y_5)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_5'] = str(text_pred_5)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_5?'] = str(text_correct_5)
        
        
        #### For 6 second validity
        
        if text_pred_6 == 0: # neutral
            text_x_6 = dead_x
            text_y_6 = dead_y
        elif text_pred_6 == 1: # neg inact
            text_x_6 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_6 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_6 == 2: # pos inact 
            text_x_6 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_6 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_6 == 3: # pos act
            text_x_6 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_6 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_6 == 4: # neg act
            text_x_6 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_6 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset

        # if text_pred_6 == target_label:
            # text_correct_6 = 1
        # else:
            # text_correct_6 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_6'] = str(text_x_6)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_6'] = str(text_y_6)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_6'] = str(text_pred_6)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_6?'] = str(text_correct_6)
        
        
        
        #### For 7 second validity
        
        if text_pred_7 == 0: # neutral
            text_x_7 = dead_x
            text_y_7 = dead_y
        elif text_pred_7 == 1: # neg inact
            text_x_7 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_7 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_7 == 2: # pos inact 
            text_x_7 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_7 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_7 == 3: # pos act
            text_x_7 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_7 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_7 == 4: # neg act
            text_x_7 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_7 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset

        
        # if text_pred_7 == target_label:
            # text_correct_7 = 1
        # else:
            # text_correct_7 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_7'] = str(text_x_7)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_7'] = str(text_y_7)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_7'] = str(text_pred_7)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_7?'] = str(text_correct_7)
        
        
        
        #### For 8 second validity
        
        if text_pred_8 == 0: # neutral
            text_x_8 = dead_x
            text_y_8 = dead_y
        elif text_pred_8 == 1: # neg inact
            text_x_8 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_8 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_8 == 2: # pos inact 
            text_x_8 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_8 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_8 == 3: # pos act
            text_x_8 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_8 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_8 == 4: # neg act
            text_x_8 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_8 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        
        # if text_pred_8 == target_label:
            # text_correct_8 = 1
        # else:
            # text_correct_8 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_8'] = str(text_x_8)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_8'] = str(text_y_8)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_8'] = str(text_pred_8)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_8?'] = str(text_correct_8)
        
        
        
        #### For 9 second validity
        
        if text_pred_9 == 0: # neutral
            text_x_9 = dead_x
            text_y_9 = dead_y
        elif text_pred_9 == 1: # neg inact
            text_x_9 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_9 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_9 == 2: # pos inact 
            text_x_9 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_9 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_9 == 3: # pos act
            text_x_9 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_9 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_9 == 4: # neg act
            text_x_9 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_9 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset

        
        # if text_pred_9 == target_label:
            # text_correct_9 = 1
        # else:
            # text_correct_9 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_9'] = str(text_x_9)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_9'] = str(text_y_9)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_9'] = str(text_pred_9)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_9?'] = str(text_correct_9)
        
        
        
        #### For 10 second validity
        
        if text_pred_10 == 0: # neutral
            text_x_10 = dead_x
            text_y_10 = dead_y
        elif text_pred_10 == 1: # neg inact
            text_x_10 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_10 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_10 == 2: # pos inact 
            text_x_10 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_10 = int(passive_end_y - (passive_end_y - origin_y)/2) + text_offset
        elif text_pred_10 == 3: # pos act
            text_x_10 = int(positive_end_x - ((positive_end_x - origin_x)/2)) + text_offset
            text_y_10 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset
        elif text_pred_10 == 4: # neg act
            text_x_10 = int(origin_x - ((origin_x - negative_end_x)/2)) + text_offset
            text_y_10 = int(origin_y - ((origin_y - active_end_y)/2)) + text_offset

        
        # if text_pred_10 == target_label:
            # text_correct_10 = 1
        # else:
            # text_correct_10 = 0
            
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_10'] = str(text_x_10)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_10'] = str(text_y_10)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_10'] = str(text_pred_10)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_10?'] = str(text_correct_10)
        

        ######### Put audio prediction visualization here
        
        
        #### For 2 second validity
        
        if audio_pred_2 == 0:
            audio_x = dead_x
            audio_y = dead_y
        elif audio_pred_2 == 1:
            audio_x = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_2 == 2:
            audio_x = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_2 == 3:
            audio_x = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_2 == 4:
            audio_x = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_2 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x'] = str(audio_x)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y'] = str(audio_y)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class'] = str(audio_pred_2)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct?'] = str(audio_correct)
        
        if audio_x != (dead_x):
            w_frame = cv2.circle(w_frame, (audio_x, audio_y), 20, audio_point_color, -1)
            
            
            
        #### For 0 second validity
        
        if audio_pred_0 == 0:
            audio_x_0 = dead_x
            audio_y_0 = dead_y
        elif audio_pred_0 == 1:
            audio_x_0 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_0 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_0 == 2:
            audio_x_0 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_0 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_0 == 3:
            audio_x_0 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_0 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_0 == 4:
            audio_x_0 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_0 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_0 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_0'] = str(audio_x_0)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_0'] = str(audio_y_0)
        
        
        
        #### For 1 second validity
        
        if audio_pred_1 == 0:
            audio_x_1 = dead_x
            audio_y_1 = dead_y
        elif audio_pred_1 == 1:
            audio_x_1 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_1 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_1 == 2:
            audio_x_1 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_1 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_1 == 3:
            audio_x_1 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_1 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_1 == 4:
            audio_x_1 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_1 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_1 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_1'] = str(audio_x_1)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_1'] = str(audio_y_1)
        
        
        
        #### For 3 second validity
        
        if audio_pred_3 == 0:
            audio_x_3 = dead_x
            audio_y_3 = dead_y
        elif audio_pred_3 == 1:
            audio_x_3 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_3 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_3 == 2:
            audio_x_3 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_3 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_3 == 3:
            audio_x_3 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_3 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_3 == 4:
            audio_x_3 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_3 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_3 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_3'] = str(audio_x_3)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_3'] = str(audio_y_3)
        
        
        
        #### For 4 second validity
        
        if audio_pred_4 == 0:
            audio_x_4 = dead_x
            audio_y_4 = dead_y
        elif audio_pred_4 == 1:
            audio_x_4 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_4 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_4 == 2:
            audio_x_4 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_4 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_4 == 3:
            audio_x_4 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_4 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_4 == 4:
            audio_x_4 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_4 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_4 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_4'] = str(audio_x_4)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_4'] = str(audio_y_4)
        
        
        
        
        #### For 5 second validity
        
        if audio_pred_5 == 0:
            audio_x_5 = dead_x
            audio_y_5 = dead_y
        elif audio_pred_5 == 1:
            audio_x_5 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_5 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_5 == 2:
            audio_x_5 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_5 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_5 == 3:
            audio_x_5 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_5 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_5 == 4:
            audio_x_5 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_5 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_5 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_5'] = str(audio_x_5)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_5'] = str(audio_y_5)
        
        
        
        #### For 6 second validity
        
        if audio_pred_6 == 0:
            audio_x_6 = dead_x
            audio_y_6 = dead_y
        elif audio_pred_6 == 1:
            audio_x_6 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_6 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_6 == 2:
            audio_x_6 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_6 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_6 == 3:
            audio_x_6 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_6 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_6 == 4:
            audio_x_6 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_6 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_6 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_6'] = str(audio_x_6)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_6'] = str(audio_y_6)
        
        
        
        #### For 7 second validity
        
        if audio_pred_7 == 0:
            audio_x_7 = dead_x
            audio_y_7 = dead_y
        elif audio_pred_7 == 1:
            audio_x_7 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_7 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_7 == 2:
            audio_x_7 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_7 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_7 == 3:
            audio_x_7 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_7 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_7 == 4:
            audio_x_7 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_7 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_7 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_7'] = str(audio_x_7)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_7'] = str(audio_y_7)
        
        
        
        #### For 8 second validity
        
        if audio_pred_8 == 0:
            audio_x_8 = dead_x
            audio_y_8 = dead_y
        elif audio_pred_8 == 1:
            audio_x_8 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_8 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_8 == 2:
            audio_x_8 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_8 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_8 == 3:
            audio_x_8 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_8 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_8 == 4:
            audio_x_8 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_8 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_8 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_8'] = str(audio_x_8)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_8'] = str(audio_y_8)
        
        
        
        
        #### For 9 second validity
        
        if audio_pred_9 == 0:
            audio_x_9 = dead_x
            audio_y_9 = dead_y
        elif audio_pred_9 == 1:
            audio_x_9 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_9 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_9 == 2:
            audio_x_9 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_9 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_9 == 3:
            audio_x_9 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_9 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_9 == 4:
            audio_x_9 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_9 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        
        # if audio_pred_9 == target_label:
            # audio_correct = 1
        # else:
            # audio_correct = 0

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_9'] = str(audio_x_9)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_9'] = str(audio_y_9)
        
        
        
        
        #### For 10 second validity
        
        if audio_pred_10 == 0:
            audio_x_10 = dead_x
            audio_y_10 = dead_y
        elif audio_pred_10 == 1:
            audio_x_10 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_10 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_10 == 2:
            audio_x_10 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_10 = int(passive_end_y - (passive_end_y - origin_y)/2) - audio_offset
        elif audio_pred_10 == 3:
            audio_x_10 = int(positive_end_x - ((positive_end_x - origin_x)/2)) - audio_offset
            audio_y_10 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        elif audio_pred_10 == 4:
            audio_x_10 = int(origin_x - ((origin_x - negative_end_x)/2)) - audio_offset
            audio_y_10 = int(origin_y - ((origin_y - active_end_y)/2)) - audio_offset
        

        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x_10'] = str(audio_x_10)
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y_10'] = str(audio_y_10)
        
        # if audio_pred_0 == target_label:
            # audio_correct_0 = 1
        # else:
            # audio_correct_0 = 0
        
        # if audio_pred_1 == target_label:
            # audio_correct_1 = 1
        # else:
            # audio_correct_1 = 0
        
        # if audio_pred_3 == target_label:
            # audio_correct_3 = 1
        # else:
            # audio_correct_3 = 0
        
        # if audio_pred_4 == target_label:
            # audio_correct_4 = 1
        # else:
            # audio_correct_4 = 0
        
        # if audio_pred_5 == target_label:
            # audio_correct_5 = 1
        # else:
            # audio_correct_5 = 0
        
        # if audio_pred_6 == target_label:
            # audio_correct_6 = 1
        # else:
            # audio_correct_6 = 0
        
        # if audio_pred_7 == target_label:
            # audio_correct_7 = 1
        # else:
            # audio_correct_7 = 0
        
        # if audio_pred_8 == target_label:
            # audio_correct_8 = 1
        # else:
            # audio_correct_8 = 0
        
        # if audio_pred_9 == target_label:
            # audio_correct_9 = 1
        # else:
            # audio_correct_9 = 0
        
        # if audio_pred_10 == target_label:
            # audio_correct_10 = 1
        # else:
            # audio_correct_10 = 0
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_0'] = str(audio_pred_0)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_0?'] = str(audio_correct_0)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_1'] = str(audio_pred_1)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_1?'] = str(audio_correct_1)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_3'] = str(audio_pred_3)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_3?'] = str(audio_correct_3)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_4'] = str(audio_pred_4)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_4?'] = str(audio_correct_4)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_5'] = str(audio_pred_5)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_5?'] = str(audio_correct_5)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_6'] = str(audio_pred_6)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_6?'] = str(audio_correct_6)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_7'] = str(audio_pred_7)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_7?'] = str(audio_correct_7)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_8'] = str(audio_pred_8)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_8?'] = str(audio_correct_8)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_9'] = str(audio_pred_9)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_9?'] = str(audio_correct_9)
        
        all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_10'] = str(audio_pred_10)
        #all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_10?'] = str(audio_correct_10)
        
        ##### Draw connecting lines between text, audio, video   ####

            
        if video_x != dead_x:
            if audio_x != (dead_x):
                w_frame = cv2.line(w_frame, (audio_x, audio_y), (video_x, video_y), (0,0,0), 2)
                
            if text_x != dead_x:
                w_frame = cv2.line(w_frame, (text_x, text_y), (video_x, video_y), (0,0,0), 2)
                
            if audio_x != (dead_x) and text_x != dead_x:
                w_frame = cv2.line(w_frame, (text_x, text_y), (audio_x, audio_y), (0,0,0), 2)
                
        else:
            if audio_x != (dead_x) and text_x != dead_x:
                w_frame = cv2.line(w_frame, (text_x, text_y), (audio_x, audio_y), (0,0,0), 2)
        
        ########### ##############################################
        
        ####### Check mismatches #######
            
        if not os.path.exists("videos\\Mismatches"):
            os.mkdir("videos\\Mismatches")
        
        # if not os.path.exists("videos\\Mismatches\\VidAudMismatches"):
            # os.mkdir("videos\\Mismatches\\VidAudMismatches")
        
        # if not os.path.exists("videos\\Mismatches\\VidTextMismatches"):
            # os.mkdir("videos\\Mismatches\\VidTextMismatches")
        
        # if not os.path.exists("videos\\Mismatches\\TextAudMismatches"):
            # os.mkdir("videos\\Mismatches\\TextAudMismatches")
        
        for i in range(0,11):
            if not os.path.exists("videos\\Mismatches\\" + str(i) + " s validity"):
                os.mkdir("videos\\Mismatches\\" + str(i) + " s validity")
            
            # if not os.path.exists("videos\\Mismatches\\VidAudMismatches\\" + str(i) + " s validity"):
                # os.mkdir("videos\\Mismatches\\VidAudMismatches\\" + str(i) + " s validity")
            # if not os.path.exists("videos\\Mismatches\\VidTextMismatches\\" + str(i) + " s validity"):
                # os.mkdir("videos\\Mismatches\\VidTextMismatches\\" + str(i) + " s validity")
                
        
        if hasFace != 0:
            
            if text_pred_2 != 0:
                if audio_pred_2 != text_pred_2:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 1
                
                elif audio_pred_2 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 1
                    
                elif text_pred_2 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 1
                    
                elif (video_pred == audio_pred_2 and video_pred == text_pred_2 and audio_pred_2 == text_pred_2):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 0
            
            
            if text_pred_0 != 0:
                if audio_pred_0 != text_pred_0:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 1
                
                elif audio_pred_0 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 1
                    
                elif text_pred_0 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 1
                    
                elif (video_pred == audio_pred_0 and video_pred == text_pred_0 and audio_pred_0 == text_pred_0):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 0
            
            if text_pred_1 != 0:
                if audio_pred_1 != text_pred_1:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 1
                
                elif audio_pred_1 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 1
                    
                elif text_pred_1 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 1
                    
                elif (video_pred == audio_pred_1 and video_pred == text_pred_1 and audio_pred_1 == text_pred_1):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 0
            
            if text_pred_3 != 0:
                if audio_pred_3 != text_pred_3:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 1
                
                elif audio_pred_3 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 1
                    
                elif text_pred_3 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 1
                    
                elif (video_pred == audio_pred_3 and video_pred == text_pred_3 and audio_pred_3 == text_pred_3):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 0
            
            if text_pred_4 != 0:
                if audio_pred_4 != text_pred_4:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 1
                
                elif audio_pred_4 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 1
                    
                elif text_pred_4 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 1
                    
                elif (video_pred == audio_pred_4 and video_pred == text_pred_4 and audio_pred_4 == text_pred_4):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 0
            
            if text_pred_5 != 0:
                if audio_pred_5 != text_pred_5:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 1
                
                elif audio_pred_5 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 1
                    
                elif text_pred_5 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 1
                    
                elif (video_pred == audio_pred_5 and video_pred == text_pred_5 and audio_pred_5 == text_pred_5):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 0
            
            if text_pred_6 != 0:
                if audio_pred_6 != text_pred_6:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 1
                
                elif audio_pred_6 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 1
                    
                elif text_pred_6 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 1
                    
                elif (video_pred == audio_pred_6 and video_pred == text_pred_6 and audio_pred_6 == text_pred_6):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 0
            
            if text_pred_7 != 0:
                if audio_pred_7 != text_pred_7:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 1
                
                elif audio_pred_7 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 1
                    
                elif text_pred_7 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 1
                    
                elif (video_pred == audio_pred_7 and video_pred == text_pred_7 and audio_pred_7 == text_pred_7):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 0
            
            if text_pred_8 != 0:
                if audio_pred_8 != text_pred_8:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 1
                
                elif audio_pred_8 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 1
                    
                elif text_pred_8 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 1
                    
                elif (video_pred == audio_pred_8 and video_pred == text_pred_8 and audio_pred_8 == text_pred_8):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 0
            
            if text_pred_9 != 0:
                if audio_pred_9 != text_pred_9:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 1
                
                elif audio_pred_9 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 1
                    
                elif text_pred_9 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 1
                    
                elif (video_pred == audio_pred_9 and video_pred == text_pred_9 and audio_pred_9 == text_pred_9):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 0
            
            if text_pred_10 != 0:
                if audio_pred_10 != text_pred_10:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 1
                
                elif audio_pred_10 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 1
                    
                elif text_pred_10 != video_pred:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 1
                    
                elif (video_pred == audio_pred_10 and video_pred == text_pred_10 and audio_pred_10 == text_pred_10):
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 0
                    
                else:
                    all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 0
        
            else:
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 0
            
                
                
        else:
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_2'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_0'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_1'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_3'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_4'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_5'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_6'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_7'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_8'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_9'] = 0
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Mismatch_10'] = 0

        
        #######################################################
        
        
        ######## Print progress #######
        
        CurrFrame = int(Capture.get(cv2.CAP_PROP_POS_FRAMES))
        percent = (CurrFrame/vidFrames) * 100
        #progress_label.setText("Completed:  " + str(round(percent,2)) + " %")
        
        a, currFileName = os.path.split(filename)
        
        print(str(n_file + 1) + " / " + str(total_files) + " : "
            + currFileName + ": " + str(int(Capture.get(cv2.CAP_PROP_POS_FRAMES))) + " / " 
            + str(int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))) + " (" + str(round(percent,2)) + " %)"
            + " \t " + "Curr time: " 
            + str(round(videoCurrDur,3)) + " \t "
            + " \t " + "Video: " + str(video_pred)
            + " \t " + "Audio (0): " + str(audio_pred_0)
            + " \t " + "Audio (2): " + str(audio_pred_2)
            + " \t " + "Text (0): " + str(text_pred_0)
            + " \t " + "Text (2): " + str(text_pred_2))
        
        
        
        w_frame = cv2.putText(w_frame, 'Activated', (2220,30), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Deactivated', (2220,700), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Positive', (2480,400), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Negative', (1980,400), font, 1.0, (25, 25, 25), 2)
        
        #w_frame = cv2.putText(w_frame, 'Target', (1950,30), font, 1.0, target_box_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Video', (1950,70), font, 1.0, video_point_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Text', (1950,110), font, 1.0, text_point_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Audio', (1950,150), font, 1.0, audio_point_color, 2)
        
        
        
        
        
        ##### Score bar #####
        
        j = CurrFrame
        
        if text_pred_2 == 1:
            this_score_text = s_negInact
        
        elif text_pred_2 == 2:
            this_score_text = s_posInact
        
        elif text_pred_2 == 3:
            this_score_text = s_posAct
        
        elif text_pred_2 == 4:
            this_score_text = s_negAct
            
        
        
        if audio_pred_2 == 1:
            this_score_aud = s_negInact
        
        elif audio_pred_2 == 2:
            this_score_aud = s_posInact
        
        elif audio_pred_2 == 3:
            this_score_aud = s_posAct
        
        elif audio_pred_2 == 4:
            this_score_aud = s_negAct
            
            
        
        if video_pred == 1:
            this_score_vid = s_negInact
        
        elif video_pred == 2:
            this_score_vid = s_posInact
        
        elif video_pred == 3:
            this_score_vid = s_posAct
        
        elif video_pred == 4:
            this_score_vid = s_negAct
            
            
        
        if hasFace == 1:
            vid_bool = True
        else:
            vid_bool = False
            
        if audio_pred_2 == 0:
            aud_bool = False
        else:
            aud_bool = True
            
        if text_pred_2 == 0:
            text_bool = False
        else:
            text_bool = True
        
        
        
        if vid_bool == False and aud_bool == False and text_bool == False:
            curr_score = 0
        
        elif vid_bool == False and aud_bool == False and text_bool == True:
            curr_score = this_score_text
        
        elif vid_bool == False and aud_bool == True and text_bool == False:
            curr_score = this_score_aud

        elif vid_bool == False and aud_bool == True and text_bool == True:
            curr_score = this_score_aud * 0.50 + this_score_text * 0.50
        
        elif vid_bool == True and aud_bool == False and text_bool == False:
            curr_score = this_score_vid

        elif vid_bool == True and aud_bool == False and text_bool == True:
            curr_score = this_score_vid * 0.50 + this_score_text * 0.50
        
        elif vid_bool == True and aud_bool == True and text_bool == False:
            curr_score = this_score_vid * 0.50 + this_score_aud * 0.50
        
        elif vid_bool == True and aud_bool == True and text_bool == True:
            curr_score = this_score_vid * w_vid + this_score_aud * w_aud + this_score_text * w_text
        
        w_frame = cv2.putText(w_frame, 'Engagement Score', (2150,850), font, 1.0, (25, 25, 25), 2)
        
        w_frame = cv2.putText(w_frame, '-1.0', (1940,900), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, '-0.5', (2100,900), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, '0.0', (2275,900), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, '0.5', (2425,900), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, '1.0', (2550,900), font, 1.0, (25, 25, 25), 2)
        
        w_frame = cv2.line(w_frame, (left_x + half_length, 950), (left_x + half_length, 930), (0,0,0), 3)  # 0.0
        
        w_frame = cv2.line(w_frame, (left_x + half_length - 150, 950), (left_x + half_length - 150, 930), (0,0,0), 3)  # -0.5
        
        w_frame = cv2.line(w_frame, (left_x, 950), (left_x, 930), (0,0,0), 3)  # -1.0
        
        w_frame = cv2.line(w_frame, (left_x + half_length + 150, 950), (left_x + half_length + 150, 930), (0,0,0), 3)  # 0.5
        
        w_frame = cv2.line(w_frame, (right_x - 5, 950), (right_x - 5, 930), (0,0,0), 3)  # 1.0
        
        
        center_x = left_x + half_length
        
        score_start_point_x = left_x
        score_start_point_y = 950
        score_end_point_x = right_x
        score_end_point_y = 1000
        score_box_color = (0, 0, 190)
        score_box_color_filled = (190, 0, 0)
        
        
        curr_score_start_point_y = 950
        curr_score_end_point_y = 1000
        
        #curr_score = -0.3
        
        #curr_score = random.uniform(-1.00, 1.00)
        if curr_score > 0.00:
            m = interp1d([-1.00,1.00], [score_start_point_x, score_end_point_x])
            curr_score_start_point_x = center_x
            curr_score_end_point_x = int(float(m(curr_score)))
            
        elif curr_score < 0.00:
            m = interp1d([-1.00,1.00], [score_start_point_x, score_end_point_x])
            curr_score_end_point_x = center_x
            curr_score_start_point_x = int(float(m(curr_score)))
            
        else:
            curr_score_start_point_x = center_x
            curr_score_end_point_x = center_x
        
        
        
        w_frame = cv2.rectangle(w_frame, (score_start_point_x, score_start_point_y), (score_end_point_x, score_end_point_y), 
                                score_box_color, -1)
                                
        
        w_frame = cv2.rectangle(w_frame, (curr_score_start_point_x, curr_score_start_point_y), (curr_score_end_point_x, curr_score_end_point_y), 
                                score_box_color_filled, -1)
                                
        
        
        ##### Score bar end #####
        
        
        
        
        
        
        small_frame = cv2.resize(w_frame, (0, 0), fx=0.25, fy=0.25)
        height, width, layers = small_frame.shape
        img_size = (width,height)
        #img_array.append(small_frame)
        
        # a_small_frame = cv2.resize(main_frame, (0, 0), fx=0.25, fy=0.25)
        # a_height, a_width, a_layers = a_small_frame.shape
        # a_img_size = (a_width,a_height)
        # framed_img_array.append(a_small_frame)
        
        # height, width, layers = w_frame.shape
        # img_size = (width,height)
        # img_array.append(w_frame)
        Image = cv2.cvtColor(w_frame, cv2.COLOR_BGR2RGB)
        #FlippedImage = cv2.flip(Image, 1)
        FlippedImage = Image
    

if __name__ == "__main__":
    
    main_folder = "videos_RQ3"

    res = []

    for file in os.listdir(main_folder):
        if file.endswith('.mp4'):
            if "processed" not in file:
                if "framed" not in file:
                    a = file.replace('.mp4','')
                    a = a + "_processed_all_data.csv"
                    if not os.path.exists(main_folder + "\\" + a):
                        res.append(file)

    #print(res)

    for i in range(0,len(res)):
        res[i] = os.path.abspath(res[i])
        a, b = os.path.split(res[i])
        res[i] = a + "\\" + main_folder + "\\" + b
        #res[i] = os.path.abspath(res[i])
        
    #print(res)
    
    print("Found " + str(len(res)) + " unprocessed files")
            
    
    processFile(res[0],0,len(res))
    
    subprocess.run(["python", "Processing_RQ3_noGUI_FINAL_noVidOutput.py"])