### No regularization
### Low learning rate


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import librosa
import numpy as np
#from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style('whitegrid')
import IPython.display as ipd
import librosa.display
import cv2
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
#from tqdm.notebook import tqdm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from livelossplot import PlotLossesKeras

import hickle as hkl

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
        
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
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.subplot(122)
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
        plt.show()
    elif plots:
        librosa.display.waveshow(data, sr=sampling_rate)

    return (data, Xdb, mfccs)

def cut_array(a, limit):
    assert len(a.shape) == 2
    if a.shape[1] > limit:
        a = a[:,:limit]
    return a


def report_res_and_plot_matrix(y_test, y_pred, plot_classes):

    #report metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    # print(f"Classes: {plot_classes}")

    #plot matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()

    tick_marks = np.arange(len(plot_classes))
    plt.xticks(ticks=tick_marks, labels=plot_classes, rotation=90)
    plt.yticks(ticks=tick_marks, labels=plot_classes, rotation=90)

    group_counts = [f'{value:0.0f}' for value in cnf_matrix.flatten()]
    group_percentages = [f'{100 * value:0.1f} %' for value in 
                       cnf_matrix.flatten()/np.sum(cnf_matrix)]
    labels = [f'{v1}\n({v2})' for v1, v2 in
            zip(group_counts,group_percentages)]
    n = int(np.sqrt(len(labels)))
    labels = np.asarray(labels).reshape(n,n)
    sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    # return metrics
    return [acc, cnf_matrix]  
main_path = 'Audio Processing\\audioData'

path0 = os.path.join(main_path, 'AudioDataset_excel.csv')

df_combined = pd.read_csv(path0)
print(df_combined.head())


############# Pre-processing ############

arr_wave, arr_spec, arr_mfccs, NN_data = [], [], [], []

for i in range(0,df_combined.shape[0]):
    print("Pickling: " + str(i) + " / " + str(df_combined.shape[0]))
    a1, a2, a3 = get_audio_data(df_combined.loc[i,'path'], calculate_db=True)
    img = np.stack((a2,) * 3,-1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    NN_data.append(grayImage)

pickle.dump(NN_data, open( os.path.join(main_path, "NN_data.pickle"), "wb" )) 


print("Pickle dumped")

############# Pre-processing end ############