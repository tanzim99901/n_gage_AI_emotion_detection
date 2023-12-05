#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg lab'
__date__ = '2020/09/22-8:50 PM'

import torchvision
from torchvision import transforms
import torch
import torch.utils.data
import core
import matplotlib
from PIL import Image
import torch.nn as nn
import torch.optim

import matplotlib.pyplot as plt
import os
import numpy as np
import utils
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    warnings.filterwarnings('ignore')
    # train_params = {
        # 'batch_size': 150,
        # 'sample_num_x':1
    # }
    
    train_params = {
        'batch_size': 150,
        'sample_num_x':1
    }
    # SETUP : recording board
    # SETUP : data setup
    # dataset_path = os.path.join(os.getcwd() + '/datasets/original/')
    # dataset_path = '/home/zhiquan/git_repositories/Emotion-Recognition/datasets/rafd/'
    
    # image_path = '/home/zhiquan/Pictures/datasets/aff'
    
    image_path = 'faceData'
    
    #labels_path = '/home/zhiquan/Pictures/datasets/aff_labels'
    
    labels_path = 'faceDataLabels'
    
    #exclude_path = '/home/zhiquan/Pictures/survey_images/labels.txt'
    train_dataset = core.FacialEmotionDataset(image_path=image_path, labels_path=labels_path, type='train', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomCrop(224, padding=10),
        transforms.RandomRotation((-20, 20)),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(
            0.8, 1.2), saturation=(0.5, 1.5), hue=0.1),
        transforms.ToTensor()
    ]))
    test_dataset = core.FacialEmotionDataset(image_path=image_path, labels_path=labels_path, type='test', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_dataset.weights, num_samples=int(train_params['sample_num_x'] * len(train_dataset)), replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_params['batch_size'],
                                               sampler=sampler, num_workers=6)
    
    
    print('{n} images loaded from training set'.format(n = len(train_loader.dataset)))
    # for t, (images, labels) in enumerate(train_loader):
        # print(images.shape)
    #print(len(train_loader[0]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=train_params['batch_size'],
                                              num_workers=6)
    print('{n} images loaded from test set'.format(n = len(test_loader.dataset)))
    #print(test_loader.shape)

    # SETUP : nn and training params
    fe_nn = core.vggs_net()
    optimizer = torch.optim.Adam(fe_nn.parameters(), lr=0.000004)
    
    # old lr = 0.0004
    record_board = SummaryWriter()
    nn_manager = core.NNManager(train_dataloader=train_loader,test_dataloader=test_loader, model=fe_nn, optimizer=optimizer, record_board=record_board, enable_gpu=True)
    # nn_manager.show_image(num=25, rand=True, on_board=True)
    print('start training')
    nn_manager.train(epochs=-1)
    nn_manager.__del__()

    print('done')
