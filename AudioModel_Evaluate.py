### No regularization
### Original model


### 76% testing, 90% training



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
        mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc = 40)
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

def getEpochNumber(s):
    return int(s[:2])

def report_res_and_plot_matrix(y_test, y_pred, plot_classes):

    #report metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    # print(f"Classes: {plot_classes}")

    #plot matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    print(cnf_matrix)
    return [acc, cnf_matrix]
    
main_path = 'Audio Processing\\audioData'

path0 = os.path.join(main_path, 'AudioDataset_excel.csv')

df_combined = pd.read_csv(path0)
print(df_combined.head())


############# Data ############

keepTest = True
checkTrain = False


# if keepTest == True:
    # X_train, X_test, y_train, y_test = train_test_split(mel_images_female, df_female.emotion2.values, test_size=0.20)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # # Our vectorized labels
    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # y_test = le.transform(y_test)
    # print(le.classes_)

    # inp_shape = (*X_train.shape[1:], 1)
    # print(inp_shape)
    
# else:
    # X_train, X_test, y_train, y_test = train_test_split(mel_images_female, df_female.emotion2.values, test_size=0.0001)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # # Our vectorized labels
    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # y_test = le.transform(y_test)
    # print(le.classes_)

    # inp_shape = (*X_train.shape[1:], 1)
    # print(inp_shape)









path4 = os.path.join(main_path, 'CNN_X_train.pickle')

with open(path4, "rb") as f:
    X_train = pickle.load(f)
    print("X_train loaded")

path4 = os.path.join(main_path, 'CNN_X_test.pickle')

with open(path4, "rb") as f:
    X_test = pickle.load(f)
    print("X_test loaded")

path4 = os.path.join(main_path, 'CNN_y_train.pickle')

with open(path4, "rb") as f:
    y_train = pickle.load(f)
    print("y_train loaded")

path4 = os.path.join(main_path, 'CNN_y_test.pickle')

with open(path4, "rb") as f:
    y_test = pickle.load(f)
    print("y_test loaded")


print("Pickle loaded")

# Our vectorized labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
print(le.classes_)

inp_shape = (*X_train.shape[1:], 1)
print(inp_shape)









## Model

pretrained_model = tf.keras.applications.DenseNet201(include_top=False, 
                                                     weights='imagenet', 
                                                     input_shape=(224,224,3))
# pretrained_model.trainable = False
for layer in pretrained_model.layers:
    if 'conv5' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

print(pretrained_model.input_shape, pretrained_model.output_shape)

transfer_model = tf.keras.models.Sequential()
transfer_model.add(pretrained_model)
transfer_model.add(tf.keras.layers.GlobalAveragePooling2D())
transfer_model.add(tf.keras.layers.Flatten())

transfer_model.add(tf.keras.layers.Dense(256))
transfer_model.add(tf.keras.layers.Dropout(0.2))

transfer_model.add(tf.keras.layers.Dense(128))
transfer_model.add(tf.keras.layers.Dropout(0.1))
transfer_model.add(tf.keras.layers.Dense(4, activation='softmax'))

transfer_model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])





res = []

for file in os.listdir(main_path):
    if file.endswith('.hdf5'):
        res.append(file)

total = len(res)        
column_names = ['Epochs', 'File_name', 'Train_acc', 'Train_prec', 'Train_rec', 'Train_F1',
'Test_acc', 'Test_prec', 'Test_rec', 'Test_F1']

out_np = np.zeros((total, len(column_names)))
out = pd.DataFrame(data = out_np, columns = column_names)

for i in range(0, len(res)):
    
    transfer_model.load_weights(main_path + "//" + res[i])
    
    ep_num = getEpochNumber(res[i])
    
    out.loc[i, 'Epochs'] = str(ep_num)
    out.loc[i, 'File_name'] = str(res[i])
    
    print("Results for " + str(ep_num) + " epochs")
    print("=======================================")

    if keepTest == True:

        if checkTrain == False:
            test_gen = DataGenerator(X_test, y_test, 16)
            print("Testing dataset results")
            print("=======================")
            #y_pred = transfer_model.predict(X_test).argmax(axis=1)
            y_pred = transfer_model.predict(test_gen).argmax(axis=1)

            print(classification_report(y_test, y_pred, target_names=le.classes_, digits=6))
            
            dict_test = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, digits=6)
            
            acc_test, conf_test = report_res_and_plot_matrix(y_test, y_pred, le.classes_)
            
            out.loc[i, 'Test_acc'] = str(acc_test)
            out.loc[i, 'Test_prec'] = str(dict_test['weighted avg']['precision'])
            out.loc[i, 'Test_rec'] = str(dict_test['weighted avg']['recall'])
            out.loc[i, 'Test_F1'] = str(dict_test['weighted avg']['f1-score'])

            #print(y_pred.shape, np.array(y_test).shape, X_test.shape)
            print("=======================")
        
        else:
            
            train_gen = DataGenerator(X_train, y_train, 16)
            test_gen = DataGenerator(X_test, y_test, 16)
            
            print("Testing dataset results")
            print("=======================")
            #y_pred = transfer_model.predict(X_test).argmax(axis=1)
            y_pred = transfer_model.predict(test_gen).argmax(axis=1)

            print(classification_report(y_test, y_pred, target_names=le.classes_, digits=6))
            
            dict_test = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, digits=6)
            
            acc_test, conf_test = report_res_and_plot_matrix(y_test, y_pred, le.classes_)
            
            out.loc[i, 'Test_acc'] = str(acc_test)
            out.loc[i, 'Test_prec'] = str(dict_test['weighted avg']['precision'])
            out.loc[i, 'Test_rec'] = str(dict_test['weighted avg']['recall'])
            out.loc[i, 'Test_F1'] = str(dict_test['weighted avg']['f1-score'])

            #print(y_pred.shape, np.array(y_test).shape, X_test.shape)
            print("=======================")
            print("")
            print("")
            print("")
            print("")
            print("")
            
            print("Training dataset results")
            print("=======================")
            #y_pred = transfer_model.predict(X_train, batch_size=16).argmax(axis=1)
            y_pred = transfer_model.predict(train_gen).argmax(axis=1)

            print(classification_report(y_train, y_pred, target_names=le.classes_, digits=6))
            
            dict_train = classification_report(y_train, y_pred, target_names=le.classes_, output_dict=True, digits=6)
            
            acc_train, conf_train = report_res_and_plot_matrix(y_train, y_pred, le.classes_)
            
            out.loc[i, 'Train_acc'] = str(acc_train)
            out.loc[i, 'Train_prec'] = str(dict_train['weighted avg']['precision'])
            out.loc[i, 'Train_rec'] = str(dict_train['weighted avg']['recall'])
            out.loc[i, 'Train_F1'] = str(dict_train['weighted avg']['f1-score'])

            #print(y_pred.shape, np.array(y_train).shape, X_train.shape)
            print("=======================")
        
    else:
        train_gen = DataGenerator(X_train, y_train, 16)
        
        #y_pred = transfer_model.predict(X_train, batch_size=16).argmax(axis=1)
        y_pred = transfer_model.predict(train_gen).argmax(axis=1)

        print(classification_report(y_train, y_pred, target_names=le.classes_, digits=6))
        
        dict_train = classification_report(y_train, y_pred, target_names=le.classes_, output_dict=True, digits=6)
        
        acc_train, conf_train = report_res_and_plot_matrix(y_train, y_pred, le.classes_)
        
        out.loc[i, 'Train_acc'] = str(acc_train)
        out.loc[i, 'Train_prec'] = str(dict_train['weighted avg']['precision'])
        out.loc[i, 'Train_rec'] = str(dict_train['weighted avg']['recall'])
        out.loc[i, 'Train_F1'] = str(dict_train['weighted avg']['f1-score'])

        #print(y_pred.shape, np.array(y_train).shape, X_train.shape)
        
out.to_excel(main_path + "//AudioModel_EvaluationResults.xlsx", index=False)