import cv2
import numpy as np
import time
import pandas as pd

import random

from scipy.interpolate import interp1d

# function called by trackbar, sets the next frame to be read
def getFrame(frame_nr):
    global Capture
    Capture.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)

#  function called by trackbar, sets the speed of playback
def setSpeed(val):
    global playSpeed
    playSpeed = max(val,1)

def setWindow(val):
    global windowSize
    windowSize = val
    
def setValidity(val):
    global validityRange
    validityRange = val

def movingAverage(arr, size):
    moving_averages = []
    
    while i < len(arr) - size + 1:
        
        # Store elements from i to i+size
        # in list to get the current window
        window = arr[i : i + size]
      
        # Calculate the average of current window
        window_average = round(sum(window) / size, 2)
          
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
          
        # Shift window to right by one position
        i += 1
      
    return moving_averages


    
img_array = []
fileFound = False
filename = ''

Init_Frame = 0
vidFrames = 0
vidLength = 0
audioCurrDur = 0
audioStartDur = 0
videoCurrDur = 0
videoStartDur = 0
processFin = False
    
    
# open video
video_dir = "videos"
filename = "fall2021_Team2_teamsession2.mp4"

processed_filename = filename.replace('.mp4','_processed.mp4')

#processed_filename = filename.replace('.mp4','_processed_resized.mp4')

dataFileName = video_dir + "\\" + filename.replace('.mp4','_processed_all_data.csv')
df = pd.read_csv(dataFileName)

# frame_list = df['Frame'].values.tolist()
# vid_x_list = df['Vid_x'].values.tolist()
# vid_y_list = df['Vid_y'].values.tolist()
# text_x_list = df['Text_x'].values.tolist()
# text_y_list = df['Text_y'].values.tolist()
# audio_x_list = df['Aud_x'].values.tolist()
# audio_y_list = df['Aud_y'].values.tolist()
# target_start_point_x_list = df['Target start point_x'].values.tolist()
# target_start_point_y_list = df['Target start point_y'].values.tolist()
# target_end_point_x_list = df['Target end point_x'].values.tolist()
# target_end_point_y_list = df['Target end point_y'].values.tolist()

# for i, var in enumerate(frame_list):
    # frame_list[i] = int(frame_list[i])
    # vid_x_list[i] = int(vid_x_list[i])
    # vid_y_list[i] = int(vid_y_list[i])
    # text_x_list[i] = int(text_x_list[i])
    # text_y_list[i] = int(text_y_list[i])
    # audio_x_list[i] = int(audio_x_list[i])
    # audio_y_list[i] = int(audio_y_list[i])
    # target_start_point_x_list[i] = int(target_start_point_x_list[i])
    # target_start_point_y_list[i] = int(target_start_point_y_list[i])
    # target_end_point_x_list[i] = int(target_end_point_x_list[i])
    # target_end_point_y_list[i] = int(target_end_point_y_list[i])

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

dead_x = origin_x
dead_y = origin_y

w_vid = 0.33
w_text = 0.33
w_aud = 0.33

s_posAct = 1
s_posInact = 0.5
s_negInact = -0.5
s_negAct = -1 

# video_point_color = (0, 0, 255) # RGB
# audio_point_color = (150, 150, 0)
# text_point_color = (0, 255, 0)
# target_box_color = (190, 0, 0)

video_point_color = (255, 0, 0)
audio_point_color = (0, 150, 150)
text_point_color = (0, 255, 0)
target_box_color = (0, 0, 190)
       
#Capture = cv2.VideoCapture(video_dir + "\\" + filename)

Capture_orig = cv2.VideoCapture(video_dir + "\\" + filename)

Capture = cv2.VideoCapture(video_dir + "\\" + processed_filename)

time.sleep(2)
fps = Capture.get(cv2.CAP_PROP_FPS)
sleep_ms = int(np.round((1/fps)*1000))
vidFrames = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))
vidLength = vidFrames/fps
print("Video FPS: " + str(fps))
print("Video Frames: " + str(vidFrames))
print("Video Duration (s): " + str(vidLength))

fps_orig = Capture_orig.get(cv2.CAP_PROP_FPS)

width_orig  = Capture_orig.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height_orig = Capture_orig.get(cv2.CAP_PROP_FRAME_HEIGHT)

#img_size_orig = (int(height_orig),int(width_orig))

img_size_orig = (int(width_orig),int(height_orig))


# create display window
cv2.namedWindow(filename,cv2.WINDOW_NORMAL)

# set wait for each frame, determines playbackspeed
playSpeed = 50
# add trackbar
cv2.createTrackbar("Frame", filename, 0,vidFrames,getFrame)
cv2.createTrackbar("Smooth", filename, 1, 100, setWindow)
cv2.createTrackbar("Validity", filename, 0, 10, setValidity)
#cv2.createTrackbar("Speed", filename, playSpeed,100,setSpeed)

windowSize = 1
validityRange = 0
# text_x_title = 'Text_x'
# text_y_title = 'Text_y'
# aud_x_title = 'Aud_x'
# aud_y_title = 'Aud_y'

text_x_title = 'Text_x_0'
text_y_title = 'Text_y_0'
aud_x_title = 'Aud_x_0'
aud_y_title = 'Aud_y_0'


#print(df.shape[0])

for i in range(0,df.shape[0]):
    #print(int(df['Frame'][i]))
    
    if int(df['Frame'][i]) == 0:
        df['Frame'][i] = i
    if int(df['Vid_x'][i]) == 0:
        df['Vid_x'][i] = df['Vid_x'][i-1]
    if int(df['Vid_y'][i]) == 0:
        df['Vid_y'][i] = df['Vid_y'][i-1]

#print(target_start_point_list)
# main loop
while 1:

    this_score_vid = 0
    this_score_aud = 0
    this_score_text = 0
        
    if windowSize < 1:
        windowSize = 1
    
    if validityRange < 0:
        validityRange = 0
    
    video_pred_str = 'Video class'
    
    if validityRange == 2:
        text_x_title = 'Text_x'
        text_y_title = 'Text_y'
        
        aud_x_title = 'Aud_x'
        aud_y_title = 'Aud_y'
        
        text_pred_str = 'Text class'
        aud_pred_str = 'Audio class'
    
    else:
        text_x_title = 'Text_x_' + str(validityRange)
        text_y_title = 'Text_y_' + str(validityRange)
        
        aud_x_title = 'Aud_x_' + str(validityRange)
        aud_y_title = 'Aud_y_' + str(validityRange)
        
        text_pred_str = 'Text class_' + str(validityRange)
        aud_pred_str = 'Audio class_' + str(validityRange)
        
    #print(text_x_title)
    
    df['New Vid_x'] = df['Vid_x'].rolling(windowSize, min_periods=1).mean()
    df['New Vid_y'] = df['Vid_y'].rolling(windowSize, min_periods=1).mean()
    
    
    # df['New Text_x'] = df['Text_x'].rolling(windowSize, min_periods=1).mean()
    # df['New Text_y'] = df['Text_y'].rolling(windowSize, min_periods=1).mean()
    # df['New Aud_x'] = df['Aud_x'].rolling(windowSize, min_periods=1).mean()
    # df['New Aud_y'] = df['Aud_y'].rolling(windowSize, min_periods=1).mean()
    
    # df['New Text_x'] = df[text_x_title].rolling(windowSize, min_periods=1).mean()
    # df['New Text_y'] = df[text_y_title].rolling(windowSize, min_periods=1).mean()
    # df['New Aud_x'] = df[aud_x_title].rolling(windowSize, min_periods=1).mean()
    # df['New Aud_y'] = df[aud_y_title].rolling(windowSize, min_periods=1).mean()
    
    df['New Text_x'] = df[text_x_title]
    df['New Text_y'] = df[text_y_title]
    df['New Aud_x'] = df[aud_x_title]
    df['New Aud_y'] = df[aud_y_title]
    
    frame_list = df['Frame'].values.tolist()
    vid_x_list = df['New Vid_x'].values.tolist()
    vid_y_list = df['New Vid_y'].values.tolist()
    text_x_list = df['New Text_x'].values.tolist()
    text_y_list = df['New Text_y'].values.tolist()
    audio_x_list = df['New Aud_x'].values.tolist()
    audio_y_list = df['New Aud_y'].values.tolist()
    target_start_point_x_list = df['Target start point_x'].values.tolist()
    target_start_point_y_list = df['Target start point_y'].values.tolist()
    target_end_point_x_list = df['Target end point_x'].values.tolist()
    target_end_point_y_list = df['Target end point_y'].values.tolist()

    for i, var in enumerate(frame_list):
        frame_list[i] = int(frame_list[i])
        vid_x_list[i] = int(vid_x_list[i])
        vid_y_list[i] = int(vid_y_list[i])
        text_x_list[i] = int(text_x_list[i])
        text_y_list[i] = int(text_y_list[i])
        audio_x_list[i] = int(audio_x_list[i])
        audio_y_list[i] = int(audio_y_list[i])
        target_start_point_x_list[i] = int(target_start_point_x_list[i])
        target_start_point_y_list[i] = int(target_start_point_y_list[i])
        target_end_point_x_list[i] = int(target_end_point_x_list[i])
        target_end_point_y_list[i] = int(target_end_point_y_list[i])
    
    
    # Get the next videoframe
    
    #ret, w_frame = Capture.read()
    
    ret, main_frame = Capture.read()

    # show frame, break the loop if no frame is found
    if ret:
        
        
        top_y = 50
        bottom_y = 650
        left_x = 2000
        right_x = 2600
        half_length = 300
        
        w_frame = main_frame[:, 0:480]
        
        w_frame = cv2.resize(w_frame,img_size_orig,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        
        

        w_frame = cv2.copyMakeBorder(w_frame, 0, 0, 0, 700, cv2.BORDER_CONSTANT, None, value = [255,255,255])
        
        w_frame = cv2.line(w_frame, (left_x + half_length, top_y), (right_x - half_length, bottom_y), (0,0,0), 9)  # The Y-axis (length 600)
        w_frame = cv2.line(w_frame, (left_x, top_y + half_length), (right_x, bottom_y - half_length), (0,0,0), 9) # The X-axis (length 600)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        # w_frame = cv2.putText(w_frame, 'Active', (2250,30), font, 1.0, (25, 25, 25), 2)
        # w_frame = cv2.putText(w_frame, 'Inactive', (2250,700), font, 1.0, (25, 25, 25), 2)
        # w_frame = cv2.putText(w_frame, 'Positive', (2480,400), font, 1.0, (25, 25, 25), 2)
        # w_frame = cv2.putText(w_frame, 'Negative', (1980,400), font, 1.0, (25, 25, 25), 2)
        
        # w_frame = cv2.putText(w_frame, 'Target', (1950,30), font, 1.0, target_box_color, 2)
        
        # w_frame = cv2.putText(w_frame, 'Video', (1950,60), font, 1.0, video_point_color, 2)
        
        # w_frame = cv2.putText(w_frame, 'Text', (1950,90), font, 1.0, text_point_color, 2)
        
        # w_frame = cv2.putText(w_frame, 'Audio', (1950,120), font, 1.0, audio_point_color, 2)

        videoCurrDur = Capture.get(cv2.CAP_PROP_POS_FRAMES) / Capture.get(cv2.CAP_PROP_FPS)
        
        target_start_point_x = target_start_point_x_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)]
        target_start_point_y = target_start_point_y_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)]         
        target_end_point_x = target_end_point_x_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)]
        target_end_point_y = target_end_point_y_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)]
        
        video_x = vid_x_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)] 
        video_y = vid_y_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)] 
        
        text_x = text_x_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)] 
        text_y = text_y_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)] 
        
        audio_x = audio_x_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)] 
        audio_y = audio_y_list[int(Capture.get(cv2.CAP_PROP_POS_FRAMES)-1)]
        
        ########## Draw target point #############
        
        w_frame = cv2.rectangle(w_frame, (target_start_point_x, target_start_point_y), (target_end_point_x, target_end_point_y), target_box_color, -1)
        
        ############# Draw video point
        
        if video_x != dead_x:
            w_frame = cv2.circle(w_frame, (video_x, video_y), 20, video_point_color, -1)
        
        # if video_x != origin_x + 10 and video_x != origin_x:
            # w_frame = cv2.circle(w_frame, (video_x, video_y), 20, video_point_color, -1)
        #w_frame = cv2.circle(w_frame, (video_x, video_y), 20, video_point_color, -1)
        
        ############# Draw text point
        
        if text_x != dead_x:
                w_frame = cv2.circle(w_frame, (text_x, text_y), 20, text_point_color, -1)

        # if text_x != int(origin_x):
            # w_frame = cv2.circle(w_frame, (text_x, text_y), 20, text_point_color, -1)

        #w_frame = cv2.circle(w_frame, (text_x, text_y), 20, text_point_color, -1)
        
        ############# Draw audio point
        
        if audio_x != (dead_x):
            w_frame = cv2.circle(w_frame, (audio_x, audio_y), 20, audio_point_color, -1)
            
            
        # if audio_x != (int(origin_x) - 10):
            # w_frame = cv2.circle(w_frame, (audio_x, audio_y), 20, audio_point_color, -1)
        
        #w_frame = cv2.circle(w_frame, (audio_x, audio_y), 20, audio_point_color, -1)
        
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
        
        # if video_x != origin_x + 10 and video_x != origin_x:
            # if audio_x != (int(origin_x) - 10):
                # w_frame = cv2.line(w_frame, (audio_x, audio_y), (video_x, video_y), (0,0,0), 2)
                
            # if text_x != int(origin_x):
                # w_frame = cv2.line(w_frame, (text_x, text_y), (video_x, video_y), (0,0,0), 2)
                
            # if audio_x != (int(origin_x) - 10) and text_x != int(origin_x):
                # w_frame = cv2.line(w_frame, (text_x, text_y), (audio_x, audio_y), (0,0,0), 2)
                
        # else:
            # if audio_x != (int(origin_x) - 10) and text_x != int(origin_x):
                # w_frame = cv2.line(w_frame, (text_x, text_y), (audio_x, audio_y), (0,0,0), 2)
        
        
        # w_frame = cv2.line(w_frame, (text_x, text_y), (video_x, video_y), (0,0,0), 2)
        # w_frame = cv2.line(w_frame, (text_x, text_y), (audio_x, audio_y), (0,0,0), 2)
        # w_frame = cv2.line(w_frame, (audio_x, audio_y), (video_x, video_y), (0,0,0), 2)
        
        
        
        w_frame = cv2.putText(w_frame, 'Activated', (2220,30), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Deactivated', (2220,700), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Positive', (2480,400), font, 1.0, (25, 25, 25), 2)
        w_frame = cv2.putText(w_frame, 'Negative', (1980,400), font, 1.0, (25, 25, 25), 2)
        
        w_frame = cv2.putText(w_frame, 'Target', (1950,30), font, 1.0, target_box_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Video', (1950,70), font, 1.0, video_point_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Text', (1950,110), font, 1.0, text_point_color, 2)
        
        w_frame = cv2.putText(w_frame, 'Audio', (1950,150), font, 1.0, audio_point_color, 2)
        
        ########### ##############################################
        
        ##### Score bar #####
        
        curr_frame = Capture.get(cv2.CAP_PROP_POS_FRAMES)
        j = curr_frame
        
        if df.loc[j,text_pred_str] == 1:
            this_score_text = s_negInact
        
        elif df.loc[j,text_pred_str] == 2:
            this_score_text = s_posInact
        
        elif df.loc[j,text_pred_str] == 3:
            this_score_text = s_posAct
        
        elif df.loc[j,text_pred_str] == 4:
            this_score_text = s_negAct
            
        
        
        if df.loc[j,aud_pred_str] == 1:
            this_score_aud = s_negInact
        
        elif df.loc[j,aud_pred_str] == 2:
            this_score_aud = s_posInact
        
        elif df.loc[j,aud_pred_str] == 3:
            this_score_aud = s_posAct
        
        elif df.loc[j,aud_pred_str] == 4:
            this_score_aud = s_negAct
            
            
        
        if df.loc[j,video_pred_str] == 1:
            this_score_vid = s_negInact
        
        elif df.loc[j,video_pred_str] == 2:
            this_score_vid = s_posInact
        
        elif df.loc[j,video_pred_str] == 3:
            this_score_vid = s_posAct
        
        elif df.loc[j,video_pred_str] == 4:
            this_score_vid = s_negAct
            
            
        
        if df.loc[j,'Has face?'] == 1:
            vid_bool = True
        else:
            vid_bool = False
            
        if df.loc[j,aud_pred_str] == 0:
            aud_bool = False
        else:
            aud_bool = True
            
        if df.loc[j,text_pred_str] == 0:
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
        
        

        
        
        #small_frame = cv2.resize(w_frame, (0, 0), fx=0.25, fy=0.25)
        
        #w_frame = w_frame[:, 2000:]
        small_frame = cv2.resize(w_frame, (1440,720))
        
        
        height, width, layers = small_frame.shape
        img_size = (width,height)
        
        Image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        FlippedImage = cv2.flip(Image, 1)
        #FlippedImage = Image
        FlippedImage = small_frame
        # ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
        # Pic = ConvertToQtFormat.scaled(1280,720, Qt.KeepAspectRatio)
        # #Pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)

        CurrFrame = int(Capture.get(cv2.CAP_PROP_POS_FRAMES))
        # percent = (CurrFrame/vidFrames) * 100
        # #print(percent)
        # progress_label.setText("Processing... " + str(round(percent,2)) + " %")
        # #print(sleep_ms)
        # time.sleep(sleep_ms/1000)
        
        #cv2.imshow(filename, w_frame)
        cv2.imshow(filename, FlippedImage)
        # update slider position on trackbar
        # NOTE: this is an expensive operation, remove to greatly increase max playback speed
        cv2.setTrackbarPos("Frame",filename, int(Capture.get(cv2.CAP_PROP_POS_FRAMES)))
        
    else:
        break

    # display frame for 'playSpeed' ms, detect key input
    key = cv2.waitKey(playSpeed)

    # stop playback when q is pressed
    if key == ord('q'):
        break

# release resources
Capture.release()
cv2.destroyAllWindows()