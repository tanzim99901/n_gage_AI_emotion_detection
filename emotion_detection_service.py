#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Xingyu Lei & Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg lab'
__date__ = '2020/11/02-8:25 PM'

import time
import zmq
import face_recognition
from random import randint
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms, utils
import cv2
import numpy as np

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
        return None
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
    crop_frame = frame[top:bottom, left:right]
    return crop_frame

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    emotion = ['None','p-a', 'u-a', 'p-i', 'u-i']
    ## < zq:init setup
    model = torch.load('test_model_alpha_0', map_location=torch.device('cpu'))
    model.eval()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # zq: init setup >
    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: %s" % message)
        # < zq: task start
        preds = -1
        frame = detect_face(video_capture)
        if frame is not None:
            img = data_transforms(frame)
            scores = model(img.unsqueeze(0))
            _, preds = scores.max(1)
            print(preds)
        # zq:task start >
        randIndex = preds+1
        currentEmotion = emotion[randIndex]

        #  Send output back to client
        b = currentEmotion.encode('utf-8')
        socket.send(b)
    # zq:relase resource
    video_capture.release()
    cv2.destroyAllWindows()
