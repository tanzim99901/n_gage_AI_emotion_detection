import fasttext
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import pickle
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import GridSearchCV
from numpy import asarray
from numpy import savetxt
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list

def create_list_empty_ints(n):
    my_list = []
    for i in range(n):
        my_list.append(0)
    return my_list
    
#model = fasttext.train_supervised('Text_Dataset\\one pred\\Text_Dataset_LATEST_train.txt')

model = fasttext.load_model("fasttext_one_pred_model.bin")

trainData = pd.read_csv('Text_Dataset\\one pred\\Text_Dataset_LATEST_train.csv')
testData = pd.read_csv('Text_Dataset\\one pred\\Text_Dataset_LATEST_test.csv')

x_train_df = trainData['text']
x_test_df = testData['text']

label_train_df = trainData['one pred']
label_test_df = testData['one pred']

k = x_train_df.values.tolist()

x_train = create_list_empty_strings(len(k))

for i in range(0,len(k)):
    x_train[i] = k[i].replace('\n','')

k = x_test_df.values.tolist()

x_test = create_list_empty_strings(len(k))

for i in range(0,len(k)):
    x_test[i] = k[i].replace('\n','')


k = label_train_df.values.tolist()

label_train = create_list_empty_ints(len(k))

for i in range(0,len(k)):
    label_train[i] = int(k[i])


k = label_test_df.values.tolist()

label_test = create_list_empty_ints(len(k))

for i in range(0,len(k)):
    label_test[i] = int(k[i])





preds = create_list_empty_ints(len(label_test))

for i in range(0,len(x_test)):
    jack = model.predict(x_test[i])
    tom = jack[0][0].replace('__label__','')
    preds[i] = int(tom) 

counter = 0
for i in range(0,len(x_test)):
    if preds[i] == label_test[i]:
        counter += 1

confMatrix_test = confusion_matrix(label_test,preds)
confMatrix_test = np.delete(confMatrix_test, 0, 0)
confMatrix_test = np.delete(confMatrix_test, 0, 1)
scores_test = precision_recall_fscore_support(label_test,preds, average='weighted')

print(confMatrix_test)
print(scores_test)

accuracy = float(confMatrix_test[0][0] + confMatrix_test[1][1] + confMatrix_test[2][2]) / np.sum(confMatrix_test)
rec = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 1)
prec = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 0)

recall = np.mean(rec)
precision = np.mean(prec)

# precision = float(confMatrix_test[0][0]) / (confMatrix_test[0][0] + confMatrix_test[0][1])
# recall = float(confMatrix_test[0][0]) / (confMatrix_test[0][0] + confMatrix_test[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1-Score: {}\n".format(f1_score))
print('=' * 89)
#print_confusion_matrix(label_test,preds)

# len(label_test
# conf = confusion_matrix(label_test,preds)
# print(conf)        

acc = (counter/len(label_test))*100

print("Testing accuracy: " + str(acc) + " %")
# print(label_test)
# print(preds)



preds = create_list_empty_ints(len(label_train))
for i in range(0,len(x_train)):
    jack = model.predict(x_train[i])
    tom = jack[0][0].replace('__label__','')
    preds[i] = int(tom) 

counter = 0
for i in range(0,len(x_train)):
    if preds[i] == label_train[i]:
        counter += 1
        
confMatrix_train = confusion_matrix(label_train,preds)
confMatrix_train = np.delete(confMatrix_train, 0, 0)
confMatrix_train = np.delete(confMatrix_train, 0, 1)
scores_train = precision_recall_fscore_support(label_train,preds, average='weighted')
acc = (counter/len(label_train))*100

print("Training accuracy: " + str(acc) + " %")


print(confMatrix_train)
print(scores_train)

accuracy = float(confMatrix_train[0][0] + confMatrix_train[1][1] + confMatrix_train[2][2]) / np.sum(confMatrix_train)
rec = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 1)
prec = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 0)

recall = np.mean(rec)
precision = np.mean(prec)

# precision = float(confMatrix_train[0][0]) / (confMatrix_train[0][0] + confMatrix_train[0][1])
# recall = float(confMatrix_train[0][0]) / (confMatrix_train[0][0] + confMatrix_train[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1-Score: {}\n".format(f1_score))
print('=' * 89)


