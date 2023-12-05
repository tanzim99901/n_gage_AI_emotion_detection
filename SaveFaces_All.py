import sys
import random
import time
import PyQt5
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QAbstractVideoBuffer, QVideoFrame, QVideoSurfaceFormat, QAbstractVideoSurface, QVideoProbe
from PyQt5.QtMultimediaWidgets import QVideoWidget
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
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import GridSearchCV
from sklearn import datasets 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import pandas as pd

import fasttext

from pydub import AudioSegment

import os

import wave

import librosa
from python_speech_features import *

import loupe_keras_tanzim as lpk

import tensorflow.compat.v1 as tf

from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import random
import itertools

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list
    
def detect_face(video_capture):
    global videoCurrDur, videoStartDur
    # Grab a single frame of video
    ret, frame = video_capture.read()

    h, w, ret = frame.shape
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_loc = face_recognition.face_locations(rgb_small_frame)
    if len(face_loc) == 0:
        return 0,0,0,0,frame,False
    else:
        tanzim = 1
    
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
    return top,bottom,left,right,frame,ret
  
video_folder = "videos"
face_folder = "faceData"
faceLabel_folder = "faceDataLabels"

# list to store files
res = []
# Iterate directory
for file in os.listdir(video_folder):
    # check only text files
    if file.endswith('.mp4'):
        if not file.endswith("_processed.mp4"):
            res.append(file)

total_vids = len(res)
        
# for i in range(0,len(res)):
    # print(res[i])
    

for i in range(0,len(res)):
    vid_df = pd.read_csv(faceLabel_folder + "\\" + "vidsDone.csv")

    vidProcessed = False
    vid_df_string = vid_df['FileName'].values.tolist()
    for k in range(0,len(vid_df_string)):
        if vid_df_string[k] == res[i]:
            vidProcessed = True
    
    if vidProcessed == False:
        n_curr_vid = i
        ######## Load transcript file 
        filename = video_folder + "\\" + res[i]
        print(filename)
        textFilename = filename.replace('.mp4','.txt')
        with open(textFilename) as f:
            textFile = f.readlines()
        #print(len(textFile))
        time_string = create_list_empty_strings(int(len(textFile)/4))
        text_string = create_list_empty_strings(int(len(textFile)/4))
        
        currText = ''
        counterText = 0
        
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
        
        ##############

        ###### Load target file #####
        
        targetFilename = filename.replace('.mp4','_target.csv')

        df_target = pd.read_csv(targetFilename)

        target_time_df = df_target['Time']
        target_df = df_target['one pred']
        target_active_df = df_target['Active score (+/-)']

        target_time_string = target_time_df.values.tolist()
        target_string = target_df.values.tolist()
        target_active_string = target_active_df.values.tolist()


        for i in range(0,len(target_time_string)):
            start = target_time_string[i][0] + target_time_string[i][1]
            end = target_time_string[i][3] + target_time_string[i][4]
            #print(start + '.........' + end)
            
            start_time = str(float(start) * 60 + 0.0001)
            end_time = str(float(end) * 60)
            
            target_time_string[i] = start_time + '>' + end_time
            #print(target_time_string[i])
        
        ################################
        time.sleep(2)

        ThreadActive = True
        # #Capture = cv2.VideoCapture("vid.mp4")
        # try:
            # print(Capture)
        # except:
            # Capture = cv2.VideoCapture(filename)
            # time.sleep(2)
            # fps = Capture.get(cv2.CAP_PROP_FPS)
            # sleep_ms = int(np.round((1/fps)*1000))
            # vidFrames = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # vidLength = vidFrames/fps
            # print("Video FPS: " + str(fps))
            # print("Video Frames: " + str(vidFrames))
            # print("Video Duration (s): " + str(vidLength))
         
            # time.sleep(2)
        
        Capture = cv2.VideoCapture(filename)
        time.sleep(2)
        fps = Capture.get(cv2.CAP_PROP_FPS)
        sleep_ms = int(np.round((1/fps)*1000))
        vidFrames = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))
        vidLength = vidFrames/fps
        print("Video FPS: " + str(fps))
        print("Video Frames: " + str(vidFrames))
        print("Video Duration (s): " + str(vidLength))
     
        time.sleep(2)
        #emotions = ['pleasant-active','unpleasant-active','pleasant-inactive','unpleasant-inactive']
        emotions = ['pos-act','neg-act','pos-inact','neg-inact']
        
        #label_df = pd.read_csv(faceLabel_folder + "\\" + "faceLabels.csv")
        
        temp_label_df = pd.DataFrame(columns = ['FileName', 'Label'])
        
        while ThreadActive:
            if int(Capture.get(cv2.CAP_PROP_POS_FRAMES)) >= vidFrames:
                df2 = pd.DataFrame({"FileName":[res[n_curr_vid]]})
                vid_df = vid_df.append(df2, ignore_index = True)
                vid_df.to_csv(faceLabel_folder + "\\" + "vidsDone.csv", index=None)
                label_df = pd.read_csv(faceLabel_folder + "\\" + "faceLabels.csv")
                label_df = label_df.append(temp_label_df, ignore_index = True)
                label_df.to_csv(faceLabel_folder + "\\" + "faceLabels.csv", index=None)
                break
            
            t,b,l,r,w_frame,ret = detect_face(Capture)

            ### Plot graph for video
            
            top_y = 50
            bottom_y = 650
            left_x = 2000
            right_x = 2600
            half_length = 300
            

            ### Get audio and video current duration
            
            #audioCurrDur = audioSource.get_pts()
            #print(audioCurrDur)
            videoCurrDur = Capture.get(cv2.CAP_PROP_POS_FRAMES) / Capture.get(cv2.CAP_PROP_FPS)
            
            
            
            
            ####### Get the target label and location #########
            
            for j in range(0,len(target_time_string)):
                beginning = float(target_time_string[j][0:target_time_string[j].index('>')])
                #print(beginning)
                end = float(target_time_string[j][target_time_string[j].index('>')+1:])
                
                if(videoCurrDur >= beginning and videoCurrDur <= end):
                    target_label = int(target_string[j])
                    target_active = float(target_active_string[j])

            if target_label == 0:
                target_start_point = (origin_x - 50, origin_y - 50)
                target_end_point = (origin_x + 50, origin_y + 50)
            elif target_label == 1:
                target_start_point = (origin_x, origin_y)
                target_end_point = (negative_end_x, passive_end_y)
            elif target_label == 2: 
                target_start_point = (origin_x, origin_y)
                target_end_point = (positive_end_x, passive_end_y)
            elif target_label == 3:
                target_start_point = (origin_x, origin_y)
                target_end_point = (positive_end_x, active_end_y)
            elif target_label == 4:
                target_start_point = (origin_x, origin_y)
                target_end_point = (negative_end_x, active_end_y)
            
            
            
            ## Target_label = 1 -> neg-inact
            ## Target_label = 2 -> pos-inact
            ## Target_label = 3 -> pos-act
            ## Target_label = 4 -> neg-act
            
            ############################
            
            
            
            #print("Capturing:" + str(Capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if type(t) is not int:
                preds_array = np.zeros((t.shape[0],), dtype=int)
                #print(preds_array)
                scores_array = np.zeros(t.shape[0])
                #print(t.shape[0])
                for idx, x in enumerate(t):
                    c_frame = w_frame[int(t[idx]):int(b[idx]), int(l[idx]):int(r[idx])]
                    #print(x)
                    if c_frame is not None:
                        if target_label != 0:
                            #print("Saving faces...")
                            imgOutName = os.path.split(filename)[1]
                            imgOutName = imgOutName.replace('.mp4','')
                            # imgOutName = imgOutName.replace('/videos/','/')
                            imgOutName = imgOutName + "_" + str(int(Capture.get(cv2.CAP_PROP_POS_FRAMES))) + "_" + str(idx) + ".jpg"
                            #print(imgOutName)
                            df2 = pd.DataFrame({"FileName":[imgOutName],"Label":[target_label]})
                            #label_df = label_df.append(df2, ignore_index = True)
                            
                            temp_label_df = temp_label_df.append(df2, ignore_index = True)
                            
                            cv2.imwrite(face_folder + "\\" + imgOutName, c_frame)
            
            percent = (int(Capture.get(cv2.CAP_PROP_POS_FRAMES))/int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))) * 100
            
            print(str(n_curr_vid + 1) + " / " + str(total_vids) + " " + str(res[n_curr_vid]) + " \t " + "Frame: " + str(int(Capture.get(cv2.CAP_PROP_POS_FRAMES))) + " / " + str(int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))) + " (" + str(round(percent,2)) + " %)" + " \t " + "Label: " + str(target_label) + " \t " + "Curr time: " + str(round(videoCurrDur,3)))
            
    
    else:
        tanzim = 1