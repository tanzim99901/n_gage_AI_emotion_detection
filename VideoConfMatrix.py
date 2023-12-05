import numpy as np



confMatrix_train = np.zeros((4,4), dtype=int)
confMatrix_train[0,0] = 4185559
confMatrix_train[0,1] = 47527
confMatrix_train[0,2] = 28331
confMatrix_train[0,3] = 250
confMatrix_train[1,0] = 218344
confMatrix_train[1,1] = 3755310
confMatrix_train[1,2] = 289375
confMatrix_train[1,3] = 1055
confMatrix_train[2,0] = 204957
confMatrix_train[2,1] = 447620
confMatrix_train[2,2] = 3605699
confMatrix_train[2,3] = 1716
confMatrix_train[3,0] = 170741
confMatrix_train[3,1] = 136822
confMatrix_train[3,2] = 110268
confMatrix_train[3,3] = 3644418

confMatrix_test = np.zeros((4,4), dtype=int)
confMatrix_test[0,0] = 80679
confMatrix_test[0,1] = 267
confMatrix_test[0,2] = 120
confMatrix_test[0,3] = 0
confMatrix_test[1,0] = 1359
confMatrix_test[1,1] = 310904
confMatrix_test[1,2] = 4697
confMatrix_test[1,3] = 36
confMatrix_test[2,0] = 1255
confMatrix_test[2,1] = 14780
confMatrix_test[2,2] = 296105
confMatrix_test[2,3] = 130
confMatrix_test[3,0] = 0
confMatrix_test[3,1] = 0
confMatrix_test[3,2] = 0
confMatrix_test[3,3] = 2


accuracy = float(confMatrix_train[0][0] + confMatrix_train[1][1] + confMatrix_train[2][2] +  + confMatrix_train[3][3]) / np.sum(confMatrix_train)
rec = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 1)
prec = np.diag(confMatrix_train) / np.sum(confMatrix_train, axis = 0)

recall = np.mean(rec)
precision = np.mean(prec)

# precision = float(confMatrix_train[0][0]) / (confMatrix_train[0][0] + confMatrix_train[0][1])
# recall = float(confMatrix_train[0][0]) / (confMatrix_train[0][0] + confMatrix_train[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Training confusion matrix: ")
print(confMatrix_train)
#print("Training scores: ")
#print(scores_test)

print("Train Accuracy: {}".format(accuracy))
print("Train Precision: {}".format(precision))
print("Train Recall: {}".format(recall))
print("Train F1-Score: {}\n".format(f1_score))
print('=' * 89)




accuracy = float(confMatrix_test[0][0] + confMatrix_test[1][1] + confMatrix_test[2][2] +  + confMatrix_test[3][3]) / np.sum(confMatrix_test)
rec = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 1)
prec = np.diag(confMatrix_test) / np.sum(confMatrix_test, axis = 0)

recall = np.mean(rec)
precision = np.mean(prec)

# precision = float(confMatrix_test[0][0]) / (confMatrix_test[0][0] + confMatrix_test[0][1])
# recall = float(confMatrix_test[0][0]) / (confMatrix_test[0][0] + confMatrix_test[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Training confusion matrix: ")
print(confMatrix_test)
#print("Training scores: ")
#print(scores_test)

print("Train Accuracy: {}".format(accuracy))
print("Train Precision: {}".format(precision))
print("Train Recall: {}".format(recall))
print("Train F1-Score: {}\n".format(f1_score))
print('=' * 89)