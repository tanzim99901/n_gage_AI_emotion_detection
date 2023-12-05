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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam

import pickle

from moviepy.editor import *

class MainWindow(QWidget):
    global found_text, found_text_2, found_text_0, found_text_1, found_text_3, found_text_4, found_text_5, found_text_6, found_text_7, found_text_8, found_text_9, found_text_10
    global textOn
    global target_time_string, target_string, target_active_string, targetFile, targetFilename, currText, counterText
    global time_string, text_string, textFilename, audioFilename, audioDirectoryname, textFile, audioFile, audioSource, audioLength
    global audioFps, audioFrames, fileFound, filename, vidFrames, vidLength, Init_Frame, slider, img_array, framed_img_array, processFin, img_size
    global PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
    global audio_model, text_model, video_model
    
    def __init__(self):
        global found_text, found_text_2, found_text_0, found_text_1, found_text_3, found_text_4, found_text_5, found_text_6, found_text_7, found_text_8, found_text_9, found_text_10
        global textOn
        global target_time_string, target_string, target_active_string, targetFile, targetFilename, audioSource
        global audioLength, audioFps, audioFrames, fileFound, filename, vidFrames, vidLength, Init_Frame, slider, img_array, framed_img_array
        global processFin, img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
        global audio_model, text_model, video_model
        super(MainWindow, self).__init__()
        
        self.setWindowIcon(QIcon("PlayerIcon.ico"))
        self.setWindowTitle("Engagement Detector")
        self.setGeometry(350,100,1280,720)
        self.setStyleSheet("background-color: rgba(255,255,255, 120);")
        
        p = self.palette()
        p.setColor(QPalette.Window, Qt.blue)
        self.setPalette(p)
        
        
        # media player
        mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videowidget = QVideoWidget()
        mediaPlayer.positionChanged.connect(self.position_changed)
        mediaPlayer.durationChanged.connect(self.duration_changed)
        
        #media player
        
        
        self.VBL = QVBoxLayout()
        
        progress_label = QLabel("Completed:  ")
        #progress_label.move(0,0)
        progress_label.setFont(QFont('Arial', 44))
        progress_label.setStyleSheet("color: yellow")
        progress_label.hide()
        
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        
        openBtn = QPushButton('Open')
        openBtn.clicked.connect(self.open_file)
        
        PauseBTN = QPushButton()
        PauseBTN.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        #PauseBTN.clicked.connect(self.CancelFeed)
        PauseBTN.clicked.connect(self.pause_video)
        PauseBTN.hide()
        
        PlayBTN = QPushButton()
        PlayBTN.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        #PlayBTN.clicked.connect(self.StartFeed)
        PlayBTN.clicked.connect(self.play_video)
        PlayBTN.hide()
        
        slider = QSlider(Qt.Horizontal)
        #slider.setRange(Init_Frame,vidFrames)
        slider.setRange(0,0)
        slider.sliderMoved.connect(self.set_position)
        slider.hide()
        
        if processFin == True:
            self.PauseBTN.show()
            self.PlayBTN.show()
            slider.show()
        
        self.VBL.addWidget(openBtn)
        self.VBL.addWidget(PauseBTN)
        self.VBL.addWidget(PlayBTN)
        self.VBL.addWidget(slider)
        self.VBL.addWidget(videowidget)
        self.VBL.addWidget(progress_label)
        
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        
        self.setLayout(self.VBL)
        
        # while processFin != True:
            # kame = 1
            
        #mediaPlayer.setVideoOutput(videowidget)
        
        
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
        
    def CancelFeed(self):
        self.Worker1.stop()
    
    def StartFeed(self):
        #self.Worker1.start()
        #print("Play video")
        self.Worker1.begin()
        self.Worker1.start()
        
    def open_file(self):
        global found_text, found_text_2, found_text_0, found_text_1, found_text_3, found_text_4, found_text_5, found_text_6, found_text_7, found_text_8, found_text_9, found_text_10
        global textOn
        global target_time_string, target_string, target_active_string, targetFile, targetFilename, currText
        global counterText, time_string, text_string, textFilename, audioFilename, audioDirectoryname, textFile, audioFile
        global audioSource, audioLength, audioFps, audioFrames, fileFound, filename, vidFrames, vidLength, img_array, framed_img_array, processFin
        global img_size, out_filename, mediaPlayer, videowidget, progress_label
        global audio_model, text_model, video_model
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        
        if filename != '':
            jack = 1
            fileFound = True
            progress_label.show()
        
        
        ######## Load transcript file 
        
        textFilename = filename.replace('.mp4','.txt')
        with open(textFilename) as f:
            textFile = f.readlines()
        #print(len(textFile))
        time_string = self.create_list_empty_strings(int(len(textFile)/4))
        text_string = self.create_list_empty_strings(int(len(textFile)/4))
        
        currText = ''
        counterText = 0
        
        for i in range(0,len(textFile)-1):
            #print(textFile[i])
            if (i%4 == 1):
                hours = textFile[i][0] + textFile[i][1]
                #print(hours)

                minutes = textFile[i][3] + textFile[i][4]
                #print(minutes)

                seconds = textFile[i][6] + textFile[i][7]
                #print(seconds)

                seconds_dec = textFile[i][9] + textFile[i][10] + textFile[i][11]
                #print(seconds_dec)

                
                start_time = str(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) + '.' + seconds_dec

                
                
                hours = textFile[i][17] + textFile[i][18]
                #print(hours)

                minutes = textFile[i][20] + textFile[i][21]
                #print(minutes)

                seconds = textFile[i][23] + textFile[i][24]
                #print(seconds)

                seconds_dec = textFile[i][26] + textFile[i][27] + textFile[i][28]
                #print(seconds_dec)


                end_time = str(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) + '.' + seconds_dec
                
               
                time_string[counterText] = start_time + '>' + end_time
                #print(time_string[counterText])
                
                text_string[counterText] = textFile[i+1]
                #print(text_string[counterText])
                counterText = counterText + 1
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
        
    # def set_position(self, position):
        # #self.mediaPlayer.setPosition(position)
        # global fileFound, filename, Capture, CurrFrame, Init_Frame, img_array, processFin, img_size, out_filename
        # Capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        # pass
        
    def position_changed(self, position):
        global audioSource, audioLength, audioFps, audioFrames, mediaPlayer, videowidget, slider, progress_label
        slider.setValue(position)
        
    def duration_changed(self, duration):
        global audioSource, audioLength, audioFps, audioFrames, mediaPlayer, videowidget, slider, progress_label
        slider.setRange(0, duration)
        
    def set_position(self, position):
        global audioSource, audioLength, audioFps, audioFrames, mediaPlayer, videowidget, slider, progress_label
        mediaPlayer.setPosition(position)
        
    def play_video(self):
        global audioSource, audioLength, audioFps, audioFrames, mediaPlayer, videowidget, slider, progress_label
        if mediaPlayer.state() != QMediaPlayer.PlayingState:
            mediaPlayer.play()
    
    def pause_video(self):
        global audioSource, audioLength, audioFps, audioFrames, mediaPlayer, videowidget, slider, progress_label
        if mediaPlayer.state() == QMediaPlayer.PlayingState:
            mediaPlayer.pause()
            
    def create_list_empty_strings(self,n):
        my_list = []
        for i in range(n):
            my_list.append('')
        return my_list

class AudioBiLSTM(nn.Module):
    def __init__(self, config):
        super(AudioBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()
        # self.init_weight()

    def init_weight(net):
        for name, param in net.named_parameters():
            if not 'ln' in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True))
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # self.lstm_net_audio = nn.LSTM(self.embedding_size,
        #                         self.hidden_dims,
        #                         num_layers=self.rnn_layers,
        #                         dropout=self.dropout,
        #                         bidirectional=self.bidirectional,
        #                         batch_first=True)
        self.lstm_net_audio = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout, batch_first=True)

        self.ln = nn.LayerNorm(self.embedding_size)

        # FC层
        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        #         h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
       # print(atten_w.shape, m.transpose(1, 2).shape)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        x = self.ln(x)
        x, _ = self.lstm_net_audio(x)
        x = x.mean(dim=1)
        out = self.fc_audio(x)
        return out

        
class Worker1(QThread):
    global all_data_out, text_data_out, target_time_string, target_string, target_active_string, targetFile, targetFilename
    global found_text, found_text_2, found_text_0, found_text_1, found_text_3, found_text_4, found_text_5, found_text_6, found_text_7, found_text_8, found_text_9, found_text_10
    global textOn
    global currText, counterText, time_string, text_string, textFilename, audioFilename, audioDirectoryname, textFile, audioFile
    global video_data_out, audioCurrDur, audioStartDur, videoCurrDur, videoStartDur, audioSource, audioLength, audioFps, audioFrames
    global fileFound, filename, Capture, CurrFrame, vidFrames, vidLength, Init_Frame, slider, fps, sleep_ms, img_array, framed_img_array, processFin
    global img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
    global main_frame, a_img_size
    global audio_model, text_model, video_model
    
    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):
        global all_data_out, text_data_out, target_time_string, target_string, target_active_string, targetFile
        global found_text, found_text_2, found_text_0, found_text_1, found_text_3, found_text_4, found_text_5, found_text_6, found_text_7, found_text_8, found_text_9, found_text_10
        global textOn
        global targetFilename, currText, counterText, time_string, text_string, textFilename, audioFilename, audioDirectoryname
        global textFile, audioFile, video_data_out, audioCurrDur, audioStartDur, videoCurrDur, videoStartDur, audioSource, audioLength
        global audioFps, audioFrames, fileFound, filename, Capture, CurrFrame, vidFrames, vidLength, Init_Frame, slider, fps
        global sleep_ms, img_array, framed_img_array, processFin, img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
        global main_frame, a_img_size
        global audio_model, text_model, video_model
        
        while fileFound != True:
            print("NO")
            
         
            
        openBtn.hide()
        
        # video_model_filename = 'test_model_alpha_0'
        # #video_model_filename = '1_11_2023_07_11_19_59_35'
        # text_model_positive_filename = "Text_RandomForest_model_Positive.pkl"
        # text_model_active_filename = "Text_RandomForest_model_Active.pkl"
        # text_vectorizer_filename = "Text_RandomForest_Vectorizer.pkl"
        
        # text_model_filename = "fasttext_one_pred_model.bin"
        # text_model = fasttext.load_model(text_model_filename)
        
        # video_model = torch.load(video_model_filename, map_location=torch.device('cpu'))
        # video_model.eval()
        # data_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
        # ])
        
        # audio_model_filename = 'Audio Processing\\audioModels_Interval_LowComplex_02\\Audio_GRU_vlad_44000.pt'
        # #audio_model_filename = 'Audio_GRU_vlad.pt'
        # audio_model = torch.load(audio_model_filename,map_location=torch.device('cpu'))
        # audio_model.eval()
        
        text_positive_pred = 0.0
        text_active_pred = 0.0
        #text_model_positive = pickle.load(open(text_model_positive_filename, 'rb'))
        #text_model_active = pickle.load(open(text_model_active_filename, 'rb'))

        #text_vectorizer = pickle.load(open(text_vectorizer_filename, 'rb'))
        
        
        
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
        
        self.ThreadActive = True
        #Capture = cv2.VideoCapture("vid.mp4")
        try:
            print(Capture)
        except:
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
            
            final_column_names = ['Frame', 'Target start point_x', 'Target start point_y',
            'Target end point_x', 'Target end point_y', 'Target class', 'Has face?',
            'Vid_x', 'Vid_y', 'Video class', 'Video correct?',
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
            'Text correct_0?', 'Text correct_1?', 'Text correct?', 'Text correct_3?', 'Text correct_4?',
            'Text correct_5?', 'Text correct_6?', 'Text correct_7?', 'Text correct_8?', 'Text correct_9?',
            'Text correct_10?',
            'Audio class_0', 'Audio class_1', 'Audio class', 'Audio class_3', 'Audio class_4',
            'Audio class_5', 'Audio class_6', 'Audio class_7', 'Audio class_8', 'Audio class_9',
            'Audio class_10', 
            'Audio correct_0?', 'Audio correct_1?', 'Audio correct?', 'Audio correct_3?', 'Audio correct_4?',
            'Audio correct_5?', 'Audio correct_6?', 'Audio correct_7?', 'Audio correct_8?', 'Audio correct_9?',
            'Audio correct_10?',
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
            audioFps = self.division(fps_num, fps_den)
            audioFrames = int(audioLength * audioFps)
            print("Audio FPS: " + str(audioFps))
            print("Audio Frames: " + str(audioFrames))
            print("Audio Duration (s): " + str(audioLength))
            audioStartDur = audioSource.get_pts()
             
            slider.setRange(Init_Frame,vidFrames)
            
        #emotions = ['pleasant-active','unpleasant-active','pleasant-inactive','unpleasant-inactive']
        #emotions = ['Negative-deactivated', 'Positive-deactivated', 'Positive-activated', 'Negative-activated']
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
        
        while self.ThreadActive:
            if int(Capture.get(cv2.CAP_PROP_POS_FRAMES)) >= vidFrames:
                self.stop()
                self.write_video()
                break
            
            t,b,l,r,w_frame,a_frame,ret = self.detect_face(Capture, audioSource)
            
            #main_frame = w_frame
            
            
            ### Plot graph for video
            
            top_y = 50
            bottom_y = 650
            left_x = 2000
            right_x = 2600
            half_length = 300
            
            #if ret:
            w_frame = cv2.copyMakeBorder(w_frame, 0, 0, 0, 700, cv2.BORDER_CONSTANT, None, value = [255,255,255])
            
            # w_frame = cv2.line(w_frame, (2300,50), (2300,650), (0,0,0), 9)  # The Y-axis (length 600)
            # w_frame = cv2.line(w_frame, (2000,350), (2600,350), (0,0,0), 9) # The X-axis (length 600)
            
            w_frame = cv2.line(w_frame, (left_x + half_length, top_y), (right_x - half_length, bottom_y), (0,0,0), 9)  # The Y-axis (length 600)
            w_frame = cv2.line(w_frame, (left_x, top_y + half_length), (right_x, bottom_y - half_length), (0,0,0), 9) # The X-axis (length 600)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            # w_frame = cv2.putText(w_frame, 'Activated', (2220,30), font, 1.0, (25, 25, 25), 2)
            # w_frame = cv2.putText(w_frame, 'Deactivated', (2220,700), font, 1.0, (25, 25, 25), 2)
            # w_frame = cv2.putText(w_frame, 'Positive', (2480,400), font, 1.0, (25, 25, 25), 2)
            # w_frame = cv2.putText(w_frame, 'Negative', (1980,400), font, 1.0, (25, 25, 25), 2)
            
            # w_frame = cv2.putText(w_frame, 'Target', (1950,30), font, 1.0, target_box_color, 2)
            
            # w_frame = cv2.putText(w_frame, 'Video', (1950,70), font, 1.0, video_point_color, 2)
            
            # w_frame = cv2.putText(w_frame, 'Text', (1950,110), font, 1.0, text_point_color, 2)
            
            # w_frame = cv2.putText(w_frame, 'Audio', (1950,150), font, 1.0, audio_point_color, 2)
            
            ################
            
            
            
            
            
            
            
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
            
            
            ############################
            
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
                    
                    
                    
                    ######

                    audio_pred_0 = self.predictAudioClass(audioDirectoryname + '//' + str(j) + '.wav', audio_model)
                    prev_audio_pred = audio_pred_0
                    ######
                    
                    # audio_pred_0 = 3
                    # prev_audio_pred = audio_pred_0
                        
                        
                        
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
            
            
            
            #print("Capturing:" + str(Capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if type(t) is not int:
                preds_array = np.zeros((t.shape[0],), dtype=int)
                #print(preds_array)
                scores_array = np.zeros(t.shape[0])
                #print(t.shape[0])
                for idx, x in enumerate(t):
                    c_frame = w_frame[int(t[idx]):int(b[idx]), int(l[idx]):int(r[idx])]
                    #a_frame = main_frame[int(t[idx]):int(b[idx]), int(l[idx]):int(r[idx])]
                    #print(x)
                    if c_frame is not None:

                        img = data_transforms(c_frame)
                        img = img.cuda()
                        scores = video_model(img.unsqueeze(0))
                        #print(scores)
                        _, preds = scores.max(1)
                        #print(_)
                        preds_array[idx] = int(preds.item())
                        scores_array[idx] = abs(_.item())
                        #print(_.item())
                        #m = interp1d([0,10],[2300,2600])
                        
                        #print(float(m(abs(_.item()))))
                        
                        
                        # if idx == 0:
                            # color = (255, 133, 233)
                            
                        # elif idx == 1:
                            # color = (255, 0, 0)
                        # elif idx == 2:
                            # color = (0, 255, 0)
                        # elif idx == 3:
                            # color = (0, 0, 255)
                        # else:
                            # color = (0, 0, 0)
                            
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

                # if video_x < origin_x and video_x >= negative_end_x and video_y > origin_y and video_y <= passive_end_y:
                    # video_pred = 1
                # elif video_x > origin_x and video_x <= positive_end_x and video_y > origin_y and video_y <= passive_end_y:
                    # video_pred = 2
                # elif video_x > origin_x and video_x <= positive_end_x and video_y < origin_y and video_y >= active_end_y:
                    # video_pred = 3
                # elif video_x < origin_x and video_x >= negative_end_x and video_y < origin_y and video_y >= active_end_y:
                    # video_pred = 4
                # else:
                    # video_pred = 0
                    
                
                if video_pred == target_label:
                    video_correct = 1
                else:
                    video_correct = 0

                hasFace = 1
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Frame'] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Has face?'] = str(hasFace)
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_x'] = str(video_x)
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_y'] = str(video_y)
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video class'] = str(video_pred)
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video correct?'] = str(video_correct)

            
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
                all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video correct?'] = str(video_correct)
                


            #print(hasFace)




# ##### OLD VIDEO METHOD START #####
                
                # #print(preds_array)
                # #print(scores_array)
                # y = np.bincount(preds_array)
                # #print(y)
                # maximum = max(y)
                # max_pred = np.bincount(preds_array).argmax()

                # loc_max_preds = np.where(preds_array == max_pred)[0]
                
                # loc_max_scores = np.zeros(loc_max_preds.shape[0])
                # for idx, x in enumerate(loc_max_preds):
                    # loc_max_scores[idx] = scores_array[x]
                
                # max_score = np.amax(loc_max_scores)

                # if max_pred == 1:
                    # if max_score > 10:
                        # max_score = 10
                        # maximum = max_score
                    # else:
                        # maximum = 10
                    # m = interp1d([0,maximum],[2300,2600])
                    # video_x = int(float(m(max_score)))
                    # video_y = int(-video_x + 2650)
                    
                    # m = interp1d([negative_end_x,positive_end_x], [min_param,max_param])
                    # video_positive_pred = int(float(m(video_x)))   

                    # m = interp1d([passive_end_y,active_end_y],[min_param,max_param])
                    # video_active_pred = int(float(m(video_y)))
                    
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)

                    
                # elif max_pred == 2:
                    # if max_score > 10:
                        # max_score = 10
                        # maximum = max_score
                    # else:
                        # maximum = 10
                    # m = interp1d([0,maximum],[2300,2000])

                    # video_x = int(float(m(max_score)))
                    # video_y = int(video_x - 1950)
                    
                    # m = interp1d([negative_end_x,positive_end_x], [min_param,max_param])
                    # video_positive_pred = int(float(m(video_x)))   
                    # #if text_active_pred is not None:
                    # m = interp1d([passive_end_y,active_end_y],[min_param,max_param])
                    # video_active_pred = int(float(m(video_y)))
                    
                    
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)
                    
                # elif max_pred == 3:
                    # if max_score > 10:
                        # max_score = 10
                        # maximum = max_score
                    # else:
                        # maximum = 10
                    # m = interp1d([0,maximum],[2300,2600])
                    
                    # video_x = int(float(m(max_score)))
                    # video_y = int(video_x - 1950)
                    
                    # m = interp1d([negative_end_x,positive_end_x], [min_param,max_param])
                    # video_positive_pred = int(float(m(video_x)))   

                    # m = interp1d([passive_end_y,active_end_y],[min_param,max_param])
                    # video_active_pred = int(float(m(video_y)))
                    
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)

                
                # elif max_pred == 4:
                    # if max_score > 10:
                        # max_score = 10
                        # maximum = max_score
                    # else:
                        # maximum = 10
                    # m = interp1d([0,maximum],[2300,2000])

                    # video_x = int(float(m(max_score)))
                    # video_y = int(-video_x + 2650)

                    
                    # m = interp1d([negative_end_x,positive_end_x], [min_param,max_param])
                    # video_positive_pred = int(float(m(video_x)))   
                    # #if text_active_pred is not None:
                    # m = interp1d([passive_end_y,active_end_y],[min_param,max_param])
                    # video_active_pred = int(float(m(video_y)))
                    
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
                    # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)

                # if video_x == 0:
                    # video_x = origin_x
                # if video_y == 0:
                    # video_y == origin_y
                
                # if video_x <= origin_x and video_y <= origin_y:
                    # video_pred = 0
                # elif video_x >= origin_x and video_y <= origin_y:
                    # video_pred = 1
                # elif video_x >= origin_x and video_y >= origin_y:
                    # video_pred = 2
                # elif video_x <= origin_x and video_y >= origin_y:
                    # video_pred = 3 
                
                # if video_pred == target_label:
                    # video_correct = 1
                # else:
                    # video_correct = 0
                
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 7] = str(video_pred)
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 10] = str(video_correct)
                
                
                # if video_x == 0:
                    # video_x = origin_x
                # if video_y == 0:
                    # video_y == origin_y
                
                # total_faces = preds_array.shape[0]
                # max_faces = 4
                
                # positive_counter = 0
                # negative_counter = 0
                # active_counter = 0
                # inactive_counter = 0
                
                # for i in range(0, preds_array.shape[0]):
                    # if preds_array[i] == 1:
                        # positive_counter = positive_counter + 1
                        # active_counter = active_counter + 1
                    # elif preds_array[i] == 2:
                        # negative_counter = negative_counter + 1
                        # active_counter = active_counter + 1
                    # elif preds_array[i] == 3:
                        # positive_counter = positive_counter + 1
                        # inactive_counter = inactive_counter + 1
                    # elif preds_array[i] == 4:
                        # negative_counter = negative_counter + 1
                        # inactive_counter = inactive_counter + 1
                        
                # positive_percent = (float(positive_counter)/float(total_faces)) * 100
                # negative_percent = (float(negative_counter)/float(total_faces)) * 100
                # active_percent = (float(active_counter)/float(total_faces)) * 100
                # inactive_percent = (float(inactive_counter)/float(total_faces)) * 100
                
                # #print('Positive: ' + str(positive_percent - negative_percent) + ' ; ' + 'Active: ' + str(active_percent - inactive_percent) + ' ; ')

            # else:
                # video_x = origin_x
                # video_y = origin_y
                
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 7] = str(0)
                # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 10] = str(0)
            
            
            
# ##### OLD VIDEO METHOD END #####
            
            
            #### NO VIDEO SECTION #####
            
            # video_x = dead_x
            # video_y = dead_y
            
            # # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 0] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
            # # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 1] = str(video_x)
            # # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 2] = str(video_y)
            # # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 7] = str(0)
            # # all_data_out.iloc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 10] = str(0)
            
            # all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Frame'] = str(Capture.get(cv2.CAP_PROP_POS_FRAMES))
            # all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_x'] = str(video_x)
            # all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Vid_y'] = str(video_y)
            # all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video class'] = str(0)
            # all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Video correct?'] = str(0)
            
            
            
            ########## Draw target point #############
            
            #w_frame = cv2.circle(w_frame, (target_loc_x, target_loc_y), 20, (0, 0, 255), -1)
            
            w_frame = cv2.rectangle(w_frame, target_start_point, target_end_point, target_box_color, -1)
            
            #w_frame = cv2.rectangle(w_frame, (origin_x - 50, origin_y - 50), (origin_x + 50, origin_y + 50), target_box_color, -1)
            
            ######################################
            
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

            if text_pred_2 == target_label:
                text_correct = 1
            else:
                text_correct = 0
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x'] = str(text_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y'] = str(text_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class'] = str(text_pred_2)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct?'] = str(text_correct)


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
            
            if text_pred_0 == target_label:
                text_correct_0 = 1
            else:
                text_correct_0 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_0'] = str(text_x_0)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_0'] = str(text_y_0)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_0'] = str(text_pred_0)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_0?'] = str(text_correct_0)
            
            
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
            
            if text_pred_1 == target_label:
                text_correct_1 = 1
            else:
                text_correct_1 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_1'] = str(text_x_1)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_1'] = str(text_y_1)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_1'] = str(text_pred_1)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_1?'] = str(text_correct_1)

            
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
            
            if text_pred_3 == target_label:
                text_correct_3 = 1
            else:
                text_correct_3 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_3'] = str(text_x_3)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_3'] = str(text_y_3)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_3'] = str(text_pred_3)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_3?'] = str(text_correct_3)
            
            
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
            
            if text_pred_4 == target_label:
                text_correct_4 = 1
            else:
                text_correct_4 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_4'] = str(text_x_4)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_4'] = str(text_y_4)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_4'] = str(text_pred_4)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_4?'] = str(text_correct_4)
            
            
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
            
            if text_pred_5 == target_label:
                text_correct_5 = 1
            else:
                text_correct_5 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_5'] = str(text_x_5)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_5'] = str(text_y_5)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_5'] = str(text_pred_5)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_5?'] = str(text_correct_5)
            
            
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

            if text_pred_6 == target_label:
                text_correct_6 = 1
            else:
                text_correct_6 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_6'] = str(text_x_6)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_6'] = str(text_y_6)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_6'] = str(text_pred_6)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_6?'] = str(text_correct_6)
            
            
            
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

            
            if text_pred_7 == target_label:
                text_correct_7 = 1
            else:
                text_correct_7 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_7'] = str(text_x_7)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_7'] = str(text_y_7)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_7'] = str(text_pred_7)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_7?'] = str(text_correct_7)
            
            
            
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
            
            if text_pred_8 == target_label:
                text_correct_8 = 1
            else:
                text_correct_8 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_8'] = str(text_x_8)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_8'] = str(text_y_8)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_8'] = str(text_pred_8)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_8?'] = str(text_correct_8)
            
            
            
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

            
            if text_pred_9 == target_label:
                text_correct_9 = 1
            else:
                text_correct_9 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_9'] = str(text_x_9)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_9'] = str(text_y_9)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_9'] = str(text_pred_9)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_9?'] = str(text_correct_9)
            
            
            
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

            
            if text_pred_10 == target_label:
                text_correct_10 = 1
            else:
                text_correct_10 = 0
                
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_x_10'] = str(text_x_10)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text_y_10'] = str(text_y_10)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text class_10'] = str(text_pred_10)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Text correct_10?'] = str(text_correct_10)
            

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
            
            if audio_pred_2 == target_label:
                audio_correct = 1
            else:
                audio_correct = 0

            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_x'] = str(audio_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Aud_y'] = str(audio_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class'] = str(audio_pred_2)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct?'] = str(audio_correct)
            
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
            
            if audio_pred_0 == target_label:
                audio_correct_0 = 1
            else:
                audio_correct_0 = 0
            
            if audio_pred_1 == target_label:
                audio_correct_1 = 1
            else:
                audio_correct_1 = 0
            
            if audio_pred_3 == target_label:
                audio_correct_3 = 1
            else:
                audio_correct_3 = 0
            
            if audio_pred_4 == target_label:
                audio_correct_4 = 1
            else:
                audio_correct_4 = 0
            
            if audio_pred_5 == target_label:
                audio_correct_5 = 1
            else:
                audio_correct_5 = 0
            
            if audio_pred_6 == target_label:
                audio_correct_6 = 1
            else:
                audio_correct_6 = 0
            
            if audio_pred_7 == target_label:
                audio_correct_7 = 1
            else:
                audio_correct_7 = 0
            
            if audio_pred_8 == target_label:
                audio_correct_8 = 1
            else:
                audio_correct_8 = 0
            
            if audio_pred_9 == target_label:
                audio_correct_9 = 1
            else:
                audio_correct_9 = 0
            
            if audio_pred_10 == target_label:
                audio_correct_10 = 1
            else:
                audio_correct_10 = 0
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_0'] = str(audio_pred_0)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_0?'] = str(audio_correct_0)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_1'] = str(audio_pred_1)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_1?'] = str(audio_correct_1)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_3'] = str(audio_pred_3)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_3?'] = str(audio_correct_3)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_4'] = str(audio_pred_4)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_4?'] = str(audio_correct_4)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_5'] = str(audio_pred_5)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_5?'] = str(audio_correct_5)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_6'] = str(audio_pred_6)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_6?'] = str(audio_correct_6)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_7'] = str(audio_pred_7)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_7?'] = str(audio_correct_7)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_8'] = str(audio_pred_8)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_8?'] = str(audio_correct_8)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_9'] = str(audio_pred_9)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_9?'] = str(audio_correct_9)
            
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio class_10'] = str(audio_pred_10)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Audio correct_10?'] = str(audio_correct_10)
            
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
                    
            
            #print(video_pred)
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
            
            


            
            ## Target data save
            target_start_point_x = target_start_point[0]
            target_start_point_y = target_start_point[1]
            target_end_point_x = target_end_point[0]
            target_end_point_y = target_end_point[1]

            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Target start point_x'] = str(target_start_point_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Target start point_y'] = str(target_start_point_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Target end point_x'] = str(target_end_point_x)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Target end point_y'] = str(target_end_point_y)
            all_data_out.loc[int(Capture.get(cv2.CAP_PROP_POS_FRAMES))-1, 'Target class'] = str(target_label)
            
            #######################################################
            
            
            ######## Print progress #######
            
            CurrFrame = int(Capture.get(cv2.CAP_PROP_POS_FRAMES))
            percent = (CurrFrame/vidFrames) * 100
            progress_label.setText("Completed:  " + str(round(percent,2)) + " %")
            
            a, currFileName = os.path.split(filename)
            
            print(currFileName + ": " + str(int(Capture.get(cv2.CAP_PROP_POS_FRAMES))) + " / " 
                + str(int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))) + " (" + str(round(percent,2)) + " %)"
                + " \t " + "Curr time: " 
                + str(round(videoCurrDur,3)) + " \t "
                + "Target: " + str(target_label)
                + " \t " + "Video: " + str(video_pred)
                + " \t " + "Audio (0): " + str(audio_pred_0)
                + " \t " + "Audio (2): " + str(audio_pred_2)
                + " \t " + "Text (0): " + str(text_pred_0)
                + " \t " + "Text (2): " + str(text_pred_2))
                
            
            w_frame = cv2.putText(w_frame, 'Activated', (2220,30), font, 1.0, (25, 25, 25), 2)
            w_frame = cv2.putText(w_frame, 'Deactivated', (2220,700), font, 1.0, (25, 25, 25), 2)
            w_frame = cv2.putText(w_frame, 'Positive', (2480,400), font, 1.0, (25, 25, 25), 2)
            w_frame = cv2.putText(w_frame, 'Negative', (1980,400), font, 1.0, (25, 25, 25), 2)
            
            w_frame = cv2.putText(w_frame, 'Target', (1950,30), font, 1.0, target_box_color, 2)
            
            w_frame = cv2.putText(w_frame, 'Video', (1950,70), font, 1.0, video_point_color, 2)
            
            w_frame = cv2.putText(w_frame, 'Text', (1950,110), font, 1.0, text_point_color, 2)
            
            w_frame = cv2.putText(w_frame, 'Audio', (1950,150), font, 1.0, audio_point_color, 2)
            
            
            
            
            small_frame = cv2.resize(w_frame, (0, 0), fx=0.25, fy=0.25)
            height, width, layers = small_frame.shape
            img_size = (width,height)
            img_array.append(small_frame)
            
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
            ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(1280,720, Qt.KeepAspectRatio)
            #Pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
            # if cv2.waitKey(sleep_ms):
                # break
            
            self.ImageUpdate.emit(Pic)
                
                
                
    def stop(self):
        global audioSource, audioLength, audioFps, audioFrames, fileFound, filename, Capture, CurrFrame, img_array, framed_img_array
        global processFin, img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
        self.ThreadActive = False
        print("Stop video")
        CurrFrame = int(Capture.get(cv2.CAP_PROP_POS_FRAMES))
        print(CurrFrame)
        #self.quit()
        
    def begin(self):
        global audioSource, audioLength, audioFps, audioFrames, fileFound, filename, Capture, CurrFrame, img_array, framed_img_array
        global processFin, img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
        self.ThreadActive = True
        print("Play video")
        print(CurrFrame)
        Capture.set(cv2.CAP_PROP_POS_FRAMES, int(CurrFrame))
        #CurrFrame = Capture.get(cv2.CAP_PROP_POS_FRAMES)
        #self.quit()
        
    def onChange(self, trackbarValue):
        global audioSource, audioLength, audioFps, audioFrames, fileFound, filename, Capture, CurrFrame, Init_Frame
        global slider, img_array, framed_img_array, processFin, img_size, PauseBTN, PlayBTN, openBtn, out_filename, mediaPlayer, videowidget, progress_label
        Capture.set(cv2.CAP_PROP_POS_FRAMES, int(trackbarValue))
        pass
        
    def detect_face(self, video_capture, audio_capture):
        global audioCurrDur, audioStartDur, videoCurrDur, videoStartDur
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
        
    def write_video(self):
        global all_data_out, text_data_out, video_data_out, audioSource, audioLength, audioFps, audioFrames, img_array, framed_img_array
        global img_size, processFin, PauseBTN, PlayBTN, slider, openBtn, out_filename, mediaPlayer, videowidget, progress_label, fps
        global main_frame, a_img_size
        openBtn.hide()
        time.sleep(1)
        
        out_filename = filename.replace('.mp4','') + '_processed.mp4'
        a_out_filename = filename.replace('.mp4','') + '_framed.mp4'
        
        print(out_filename)
        
        all_data_out.to_csv(out_filename.replace('.mp4','') + "_all_data.csv", index=False)
        
        out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc(*'XVID'),fps, img_size)
 
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        # a_out = cv2.VideoWriter(a_out_filename,cv2.VideoWriter_fourcc(*'XVID'),fps, a_img_size)
 
        # for i in range(len(framed_img_array)):
            # a_out.write(framed_img_array[i])
        # a_out.release()
        
        self.MixAudioVideo(out_filename, filename.replace('.mp4','.wav'), filename.replace('.mp4','_processed_wSound.mp4'))

        progress_label.hide()
        
        time.sleep(1)
        mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(out_filename)))
        PauseBTN.show()
        PlayBTN.show()
        slider.show()
        processFin = True
    
    def MixAudioVideo(self, video_path, audio_path, output_path):
        video = VideoFileClip(video_path)
        fps = video.fps
        #print(fps)
        video = video.set_audio(AudioFileClip(audio_path))
        video.write_videofile(output_path)    

    # Function to find division without 
    # using '/' operator
    def division(self, num1, num2):
         
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
        
    def create_list_empty_strings(self,n):
        my_list = []
        for i in range(n):
            my_list.append('')
        return my_list
        
    def wav2vlad(self,wave_data,sr):
        global cluster_size
        signal = wave_data
        #signal, sr = librosa.load(wave_file)
        melspec = librosa.feature.melspectrogram(y=signal, n_mels=80,sr=sr).astype(np.float32).T
        melspec = np.log(np.maximum(1e-6, melspec))
        
        feature_size = melspec.shape[1]
        max_samples = melspec.shape[0]
        output_dim = cluster_size * 16
        feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                                cluster_size=cluster_size, output_dim=output_dim) \
                                    (tf.convert_to_tensor(melspec))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            r = feat.numpy()
        return r
        
    def get_audio_data(self, path, calculate_db=False, calculate_mfccs=False, plots=False):
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
        
    def predictAudioClass(self, file_loc, model):
        NN_data = []
        a1, a2, a3 = self.get_audio_data(file_loc, calculate_db=True)
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
    

if __name__ == "__main__":

    global audio_model, text_model, video_model
    ##### Load models start #####
    # pretrained_model = tf.keras.applications.DenseNet201(include_top=False, 
                                                         # weights='imagenet', 
                                                         # input_shape=(224,224,3))
    # for layer in pretrained_model.layers:
        # if 'conv5' in layer.name:
            # layer.trainable = True
        # else:
            # layer.trainable = False
    # audio_model = tf.keras.models.Sequential()
    # audio_model.add(pretrained_model)
    # audio_model.add(tf.keras.layers.GlobalAveragePooling2D())
    # audio_model.add(tf.keras.layers.Flatten())
    # audio_model.add(tf.keras.layers.Dense(256))
    # audio_model.add(tf.keras.layers.Dropout(0.2))
    # audio_model.add(tf.keras.layers.Dense(128))
    # audio_model.add(tf.keras.layers.Dropout(0.1))
    # audio_model.add(tf.keras.layers.Dense(4, activation='softmax'))
    # audio_model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # audio_model_weights_filename = 'Audio_Model_CNN.hdf5'
    # audio_model.load_weights(audio_model_weights_filename)
    
    
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
    audio_model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    filename = ''
    cluster_size = 16
    Init_Frame = 0
    vidFrames = 0
    vidLength = 0
    audioCurrDur = 0
    audioStartDur = 0
    videoCurrDur = 0
    videoStartDur = 0
    processFin = False
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())