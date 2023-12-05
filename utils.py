#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg lab'
__date__ = '2020/09/24-5:32 PM'

import io
import itertools

import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
def pred_info(preds, labels):
    warnings.filterwarnings('ignore')
    return {'correct_num': preds.argmax(dim=1).eq(labels).sum().item()}


def confusion_mat(preds:torch.Tensor, labels:torch.Tensor, class_num):
    warnings.filterwarnings('ignore')
    preds =preds.cpu()
    labels = labels.cpu()
    tmp_cm = np.zeros((class_num,class_num))
    for i in range(len(labels)):
        tmp_cm[int(labels[i].numpy())][int(preds[i].numpy())] += 1.0
    return tmp_cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    warnings.filterwarnings('ignore')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig