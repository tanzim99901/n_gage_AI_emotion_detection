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
    
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def print_confusion_matrix(y_test, y_pred):
    #output = np.zeros([config['num_classes'],config['num_classes']], dtype=int)
    output = np.zeros([5,5], dtype=int)

    for i in range(0, len(y_test)):
        #if y_test.numpy()[i] == 0:
        if np.array(y_test)[i] == 0:
            if np.array(y_pred)[i] == 0:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 1:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 2:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 3:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 4:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            
        elif np.array(y_test)[i] == 1:
            if np.array(y_pred)[i] == 0:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 1:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 2:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 3:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 4:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            
        elif np.array(y_test)[i] == 2:
            if np.array(y_pred)[i] == 0:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 1:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 2:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 3:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 4:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            
        elif np.array(y_test)[i] == 3:
            if np.array(y_pred)[i] == 0:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 1:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 2:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 3:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 4:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            
        elif np.array(y_test)[i] == 4:
            if np.array(y_pred)[i] == 0:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 1:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 2:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 3:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            elif np.array(y_pred)[i] == 4:
                output[np.array(y_test)[i]][np.array(y_pred)[i]] = output[np.array(y_test)[i]][np.array(y_pred)[i]] + 1
            
    print("Confusion matrix")
    print(output)
    #np.savetxt(main_folder + "//confusion_matrix.csv", output, delimiter=",")
    np.savetxt("text_confusion_matrix.csv", output, delimiter=",")
    return output    




params = {
    "lr" : 0.001,
    "epoch" : 500,
    "lrUpdateRate" : 10,
    }


print(len(params))


total = len(params)        
column_names = ['Epochs', 'Learning rate', 'LR Update rate', 'File_name', 'Train_acc', 'Train_prec', 'Train_rec', 'Train_F1',
'Test_acc', 'Test_prec', 'Test_rec', 'Test_F1']

#out_np = np.zeros((total, len(column_names)))
dataFile = pd.DataFrame(columns = column_names)

dataFile_loc = "Text_Dataset\\trainingPerformance.csv"

dataFile.to_csv(dataFile_loc, index=False)

iterations_lr = 6
iterations_epoch = 16
iterations_lr_update = 3

for p in range(0, iterations_lr):
    for q in range(0, iterations_epoch):
        for r in range(0, iterations_lr_update):
        
        
            out = pd.read_csv(dataFile_loc)
            
            
            newLR = params['lr'] - p * 0.0001
            newEpochs = params['epoch'] + q * 100
            newLR_UpdateRate = params['lrUpdateRate'] - r * 4
            print("Training model with " + str(newLR) + " learning rate and "
                  + str(newEpochs) + " epochs and " + str(newLR_UpdateRate) + " rate of LR update")
            


            model = fasttext.train_supervised('Text_Dataset\\one pred\\Text_Dataset_LATEST_train.txt',
                                                lr=newLR,
                                                epoch=newEpochs,
                                                lrUpdateRate=newLR_UpdateRate,
                                                verbose=2)

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

            accuracy_test = float(confMatrix_test[0][0] + confMatrix_test[1][1] + confMatrix_test[2][2]) / np.sum(confMatrix_test)
            rec_test = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 1)
            prec_test = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 0)

            recall_test = np.mean(rec_test)
            precision_test = np.mean(prec_test)

            f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

            print("Accuracy_test: {}".format(accuracy_test))
            print("Precision_test: {}".format(precision_test))
            print("Recall_test: {}".format(recall_test))
            print("F1-Score_test: {}\n".format(f1_score_test))
            print('=' * 89)
                   

            acc_test = (counter/len(label_test))*100

            print("Testing accuracy: " + str(acc_test) + " %")



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
            acc_train = (counter/len(label_train))*100

            print("Training accuracy: " + str(acc_train) + " %")


            print(confMatrix_train)
            print(scores_train)

            accuracy_train = float(confMatrix_train[0][0] + confMatrix_train[1][1] + confMatrix_train[2][2]) / np.sum(confMatrix_train)
            rec_train = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 1)
            prec_train = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 0)

            recall_train = np.mean(rec_train)
            precision_train = np.mean(prec_train)

            f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

            print("Accuracy: {}".format(accuracy_train))
            print("Precision: {}".format(precision_train))
            print("Recall: {}".format(recall_train))
            print("F1-Score: {}\n".format(f1_score_train))
            print('=' * 89)



            #print_results(*model.test('Text_Dataset\\one pred\\Text_Dataset_LATEST_test.txt'))
            
            modelName = ("fasttext_one_pred_model_" + str(round(newLR, 4)) + "_" + str(newEpochs) + "_"
                        + str(newLR_UpdateRate) + "_"
                        + str(round(accuracy_test,3)) + ".bin")
            
            model.save_model("Text_Dataset" + "\\" + modelName)

            #print_results(*model.test('Text_Dataset\\one pred\\Text_Dataset_LATEST_train.txt'))

            # out.loc[i, 'Epochs'] = str(ep_num)
            # out.loc[i, 'File_name'] = str(res[i])

            df1 = pd.DataFrame({'Epochs':[str(newEpochs)],
                                'Learning rate':[str(newLR)],
                                'LR Update rate':[str(newLR_UpdateRate)],
                                'File_name':[modelName],
                                'Train_acc':[str(accuracy_train)],
                                'Train_prec':[str(precision_train)],
                                'Train_rec':[str(recall_train)],
                                'Train_F1':[str(f1_score_train)],
                                'Test_acc':[str(accuracy_test)],
                                'Test_prec':[str(precision_test)],
                                'Test_rec':[str(recall_test)],
                                'Test_F1':[str(f1_score_test)]
                                })
            out = out.append(df1, ignore_index = True)
            
            out.to_csv(dataFile_loc, index=False)