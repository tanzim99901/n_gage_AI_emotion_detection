#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg lab'
__date__ = '2020/11/02-9:07 PM'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = ' Xingyu Lei & Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg-lab'
__date__ = '2020/11/02-11:40 AM'

import face_recognition
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms, utils


def detect_face(video_capture):
    # Grab a single frame of video
    _, frame = video_capture.read()
    h, w, _ = frame.shape
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_loc = face_recognition.face_locations(rgb_small_frame)
    if len(face_loc) == 0:
        return(0,0,0,0), None,None
    else:
        face_loc = face_loc[0]
    top, right, bottom, left = face_loc
    h_len = bottom - top
    w_len = right - left
    new_len = max(h_len, w_len) * 1.5
    center = np.array([top + h_len / 2, left + w_len / 2])
    scale = 4
    top = (int)(center[0] - new_len / 2) * scale
    bottom = (int)(center[0] + new_len / 2) * scale
    left = (int)(center[1] - new_len / 2) * scale
    right = (int)(center[1] + new_len / 2) * scale
    top = max(0, top)
    bottom = min(h, bottom)
    left = max(0, left)
    right = min(w, right)
    # Display the resulting image
    crop_frame = frame[top:bottom, left:right]
    print(crop_frame.shape)
    # Draw a label with a name below the face
    # Draw a box around the face


    #



    return (top,bottom,left,right),crop_frame,frame


if __name__ == '__main__':
    model = torch.load('test_model_alpha_0', map_location=torch.device('cpu'))
    model.eval()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Get a reference to webcam #0 (the default one)
    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture("vid.mp4")
    emotions = ['pleasant-active','unpleasant-active','pleasant-inactive','unpleasant-inactive']
    # Find all the faces and face encodings in the current frame of video
    while True:
        (t,b,l,r),c_frame,w_frame = detect_face(video_capture)
        # if c_frame is not None:

            # img = data_transforms(c_frame)
            # scores = model(img.unsqueeze(0))
            # _, preds = scores.max(1)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.rectangle(w_frame, (l, t), (r, b), (233, 233, 233), 2)
            # cv2.rectangle(w_frame, (l, b - 35), (r, b), (233, 233, 233), cv2.FILLED)
            # cv2.putText(w_frame, emotions[preds], (l + 6, b - 6), font, 1.0, (25, 25, 25), 2)
            # # # Draw a box around the face
            # cv2.imshow('Video', w_frame)
        # # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
