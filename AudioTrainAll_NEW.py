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

from tensorflow.keras.utils import Sequence

import hickle as hkl

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


# ############# Pre-processing (disabled) ############

# arr_wave, arr_spec, arr_mfccs, NN_data = [], [], [], []

# for i in range(0,df_combined.shape[0]):
    # print("Pickling: " + str(i) + " / " + str(df_combined.shape[0]))
    # a1, a2, a3 = get_audio_data(df_combined.loc[i,'path'], calculate_db=True)
    # img = np.stack((a2,) * 3,-1)
    # img = img.astype(np.uint8)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayImage = cv2.resize(grayImage, (224, 224))
    # NN_data.append(grayImage)

# pickle.dump(NN_data, open( os.path.join(main_path, "NN_data.pickle"), "wb" )) 


# print("Pickle dumped")

# ############# Pre-processing (disabled) end ############








#### Load pickle file

path4 = os.path.join(main_path, 'NN_data.pickle')

with open(path4, "rb") as f:
    NN_data = pickle.load(f)
    print("File 4 of 4 loaded")


print("Pickle loaded")





mel_images = np.array(NN_data)
print(mel_images.shape)

rgb_batch = np.repeat(mel_images[..., np.newaxis], 3, -1)
print(rgb_batch.shape)  # (4720, 224, 224, 3)



################## repeat in the case we start from this section
female_idxs = [i[0] for i in enumerate(df_combined.actors.values) if "student" in i[1]]
df_female = df_combined.loc[df_combined.actors == 'student']
##################

mel_images_female = rgb_batch[female_idxs,:,:,:]

keepTest = True
checkTrain = True


if keepTest == True:
    X_train, X_test, y_train, y_test = train_test_split(mel_images_female, df_female.emotion2.values, test_size=0.20)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    pickle.dump(X_train, open(os.path.join(main_path, "CNN_X_train.pickle"), "wb" ))
    pickle.dump(X_test, open(os.path.join(main_path, "CNN_X_test.pickle"), "wb" ))
    pickle.dump(y_train, open(os.path.join(main_path, "CNN_y_train.pickle"), "wb" ))
    pickle.dump(y_test, open(os.path.join(main_path, "CNN_y_test.pickle"), "wb" )) 

    # Our vectorized labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    print(le.classes_)

    inp_shape = (*X_train.shape[1:], 1)
    print(inp_shape)
    
else:
    X_train, X_test, y_train, y_test = train_test_split(mel_images_female, df_female.emotion2.values, test_size=0.001)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    pickle.dump(X_train, open(os.path.join(main_path, "CNN_X_train.pickle"), "wb" ))
    pickle.dump(X_test, open(os.path.join(main_path, "CNN_lowLR_X_test.pickle"), "wb" ))
    pickle.dump(y_train, open(os.path.join(main_path, "CNN_lowLR_y_train.pickle"), "wb" ))
    pickle.dump(y_test, open(os.path.join(main_path, "CNN_lowLR_y_test.pickle"), "wb" )) 

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
#transfer_model.add(tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L1(0.01)))
transfer_model.add(tf.keras.layers.Dropout(0.2))

transfer_model.add(tf.keras.layers.Dense(128))
transfer_model.add(tf.keras.layers.Dropout(0.1))
transfer_model.add(tf.keras.layers.Dense(4, activation='softmax'))
# model.add(tf.keras.layers.Activation('softmax'))

#transfer_model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

transfer_model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#transfer_model.compile(optimizer=Adam(lr=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# set callbacks
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                 # factor=0.5, patience=4, 
                                                 # verbose=1, mode='max', 
                                                 # min_lr=0.00001)
                                                 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                 factor=0.1, patience=10, 
                                                 verbose=1, mode='max', 
                                                 min_lr=0.0000001)
                                                 
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                 # factor=0.1, patience=10, 
                                                 # verbose=1, mode='max', 
                                                 # min_lr=0.00000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, 
                                              verbose=1)

# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(main_path,'weights_.hdf5'), 
                                                      # save_weights_only=True, 
                                                      # monitor='val_accuracy', 
                                                      # mode='max', 
                                                      # save_best_only=True,
                                                      # period=1)
                                                      
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(main_path,'{epoch:02d}_weights_CNN-{val_accuracy:.2f}.hdf5'), 
                                                      save_weights_only=True, 
                                                      monitor='val_accuracy', 
                                                      mode='max',
                                                      period=10)
                                                      

tb_callback = tf.keras.callbacks.TensorBoard('./logs_CNN', update_freq=1)

transfer_model.summary()


print(transfer_model.input_shape, transfer_model.output_shape)

# thistory = transfer_model.fit(X_train, y_train, batch_size=16, epochs=500, validation_split=0.1, 
                    # callbacks=[early_stop, model_checkpoint, reduce_lr, tb_callback])
                    
# thistory = transfer_model.fit(X_train, y_train, batch_size=16, epochs=500, validation_split=0.1, 
                    # callbacks=[model_checkpoint, reduce_lr, tb_callback])
                    
thistory = transfer_model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), 
                    callbacks=[model_checkpoint, reduce_lr, tb_callback])
                    


transfer_model.save_weights('weights_CNN.hdf5')


transfer_model.load_weights('weights_CNN.hdf5')
# transfer_model.load_weights(os.path.join(main_path,'weights_.hdf5'))

if keepTest == True:

    if checkTrain == False:
        print("Testing dataset results")
        print("=======================")
        y_pred = transfer_model.predict(X_test).argmax(axis=1)

        print(classification_report(y_test, y_pred, target_names=le.classes_))
        tparams = report_res_and_plot_matrix(y_test, y_pred, le.classes_)

        print(y_pred.shape, np.array(y_test).shape, X_test.shape)
        print("=======================")
    
    else:
        
        train_gen = DataGenerator(X_train, y_train, 16)
        
        print("Training dataset results")
        print("=======================")
        #y_pred = transfer_model.predict(X_train).argmax(axis=1)
        y_pred = transfer_model.predict(train_gen).argmax(axis=1)

        print(classification_report(y_train, y_pred, target_names=le.classes_))
        tparams = report_res_and_plot_matrix(y_train, y_pred, le.classes_)

        print(y_pred.shape, np.array(y_train).shape, X_train.shape)
        print("=======================")
        print("")
        print("")
        print("")
        print("")
        print("")
        
        
        print("Testing dataset results")
        print("=======================")
        y_pred = transfer_model.predict(X_test).argmax(axis=1)

        print(classification_report(y_test, y_pred, target_names=le.classes_))
        tparams = report_res_and_plot_matrix(y_test, y_pred, le.classes_)

        print(y_pred.shape, np.array(y_test).shape, X_test.shape)
        print("=======================")
        
        
        
    
else:
    train_gen = DataGenerator(X_train, y_train, 16)
    y_pred = transfer_model.predict(train_gen).argmax(axis=1)

    print(classification_report(y_train, y_pred, target_names=le.classes_))
    tparams = report_res_and_plot_matrix(y_train, y_pred, le.classes_)

    print(y_pred.shape, np.array(y_train).shape, X_train.shape)









# summarize history for accuracy
plt.plot(thistory.history['accuracy'])
plt.plot(thistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(thistory.history['loss'])
plt.plot(thistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()