import os
import pandas as pd
from scipy.interpolate import interp1d
import math
import fasttext
import numpy as np
import shutil
import subprocess
def create_list_empty_strings(n):
    my_list = []
    for i in range(n):
        my_list.append('')
    return my_list

keep_cat4 = True    
video_folder = "videos"
face_folder = "faceData"
#face_folder = "faceData_1"
faceLabel_folder = "faceDataLabels"

df = pd.read_csv("faceDataLabels\\" + "faceLabels.csv")

if os.path.exists(faceLabel_folder + "\\train.txt"):
    os.remove(faceLabel_folder + "\\train.txt")

if os.path.exists(faceLabel_folder + "\\test.txt"):
    os.remove(faceLabel_folder + "\\test.txt")
    
if os.path.exists(faceLabel_folder + "\\faceLabels.txt"):
    os.remove(faceLabel_folder + "\\faceLabels.txt")



df_temp = pd.DataFrame(columns = ['FileName', 'Label'])

df_cat1 = df[df['Label'] == 1]

df_cat1 = df_cat1.reset_index(drop=True)

df_cat2 = df[df['Label'] == 2]

df_cat2 = df_cat2.reset_index(drop=True)

df_cat3 = df[df['Label'] == 3]

df_cat3 = df_cat3.reset_index(drop=True)

df_cat4 = pd.DataFrame(columns = ['FileName', 'Label'])

df_train = pd.DataFrame(columns = ['FileName', 'Label'])

df_test = pd.DataFrame(columns = ['FileName', 'Label'])

df_total = pd.DataFrame(columns = ['FileName', 'Label'])

split_ratio = 0.8

train_df_cat1 = df_cat1.sample(frac = split_ratio)
test_df_cat1 = df_cat1.drop(train_df_cat1.index)

train_df_cat2 = df_cat2.sample(frac = split_ratio)
test_df_cat2 = df_cat2.drop(train_df_cat2.index)

train_df_cat3 = df_cat3.sample(frac = split_ratio)
test_df_cat3 = df_cat3.drop(train_df_cat3.index)






if keep_cat4 == True:
    tom = "Team12_teamsession2_43250_1.jpg"


    jack = tom.replace('.jpg','')
    jack = jack.replace('Team','')
    jack = jack.replace('teamsession','')
    jack = jack.replace('_','')
    jack = int(jack)

    copy_number = np.zeros((10,), dtype=int)
    copy_name = create_list_empty_strings(len(copy_number))
    copy_label = 4

    res = []

    for file in os.listdir(face_folder):
        if file == tom:
            res.append(file)

    for i in range(0,len(copy_number)):
        copy_number[i] = jack + i
        copy_name[i] = "Team12_teamsession2_" + str(copy_number[i]) + "_1.jpg"
        shutil.copyfile(face_folder + "\\" + tom, face_folder + "\\" + copy_name[i])
        df1 = pd.DataFrame({"FileName":[copy_name[i]],"Label":[copy_label]})
        df_cat4 = df_cat4.append(df1, ignore_index = True)

    print(tom)
    print(jack)
    print(copy_number)
    print(copy_name)

    train_df_cat4 = df_cat4.sample(frac = split_ratio)
    test_df_cat4 = df_cat4.drop(train_df_cat4.index)
    train_df_cat4 = train_df_cat4.reset_index(drop=True)
    test_df_cat4 = test_df_cat4.reset_index(drop=True)
    df_train = df_train.append(train_df_cat4, ignore_index = True)
    df_test = df_test.append(test_df_cat4, ignore_index = True)

train_df_cat1 = train_df_cat1.reset_index(drop=True)
train_df_cat2 = train_df_cat2.reset_index(drop=True)
train_df_cat3 = train_df_cat3.reset_index(drop=True)


test_df_cat1 = test_df_cat1.reset_index(drop=True)
test_df_cat2 = test_df_cat2.reset_index(drop=True)
test_df_cat3 = test_df_cat3.reset_index(drop=True)


df_train = df_train.append(train_df_cat1, ignore_index = True)
df_train = df_train.append(train_df_cat2, ignore_index = True)
df_train = df_train.append(train_df_cat3, ignore_index = True)


df_test = df_test.append(test_df_cat1, ignore_index = True)
df_test = df_test.append(test_df_cat2, ignore_index = True)
df_test = df_test.append(test_df_cat3, ignore_index = True)


df_total = df_total.append(df_train, ignore_index = True)
df_total = df_total.append(df_test, ignore_index = True)

df_train_txt = df_train["FileName"]
df_test_txt = df_test["FileName"]
df_total_txt = df_total["FileName"]

print(df_train.shape[0])
print(df_test.shape[0])
print(df_total.shape[0])
print(df.shape[0])



for i in range(0,df_train.shape[0]):
    print("Building train dataset : " + str(i) + " / " + str(df_train.shape[0]))
    inter = str(df_train.loc[i,"FileName"]) + " " + str(df_train.loc[i,"Label"] - 1)
    df_train_txt.iloc[i] = inter

df_train_txt.to_csv(faceLabel_folder + "\\" + "train.txt", header=None, index=None, sep='\n', mode='a')

for i in range(0,df_test.shape[0]):
    print("Building test dataset : " + str(i) + " / " + str(df_test.shape[0]))
    inter = str(df_test.loc[i,"FileName"]) + " " + str(df_test.loc[i,"Label"] - 1)
    df_test_txt.iloc[i] = inter

df_test_txt.to_csv(faceLabel_folder + "\\" + "test.txt", header=None, index=None, sep='\n', mode='a')

for i in range(0,df_total.shape[0]):
    print("Building full dataset : " + str(i) + " / " + str(df_total.shape[0]))
    inter = str(df_total.loc[i,"FileName"]) + " " + str(df_total.loc[i,"Label"] - 1)
    df_total_txt.iloc[i] = inter

df_total_txt.to_csv(faceLabel_folder + "\\" + "faceLabels.txt", header=None, index=None, sep='\n', mode='a')

####### Uncomment if you want to run the Video Training after this code ends ######

#subprocess.run(["python", "VideoTraining.py"])


























# # # list to store files
# # res = []
# # # Iterate directory
# # for file in os.listdir(video_folder):
    # # # check only text files
    # # if file.endswith('.mp4'):
        # # if not file.endswith("_processed.mp4"):
            # # res.append(file)

# # total_vids = len(res)

# # n_curr_vid = 10
# # vid_df = pd.read_csv(faceLabel_folder + "\\" + "vidsDone.csv")
# # df2 = pd.DataFrame({"FileName":[res[n_curr_vid]]})
# # vid_df = vid_df.append(df2, ignore_index = True)
# # vid_df.to_csv(faceLabel_folder + "\\" + "vidsDone.csv", index=None)

# # for i in range(0,len(res)):
    # # vid_df = pd.read_csv(faceLabel_folder + "\\" + "vidsDone.csv")

    # # vidProcessed = False
    # # vid_df_string = vid_df['FileName'].values.tolist()
    # # for k in range(0,len(vid_df_string)):
        # # if vid_df_string[k] == res[i]:
            # # vidProcessed = True
    
    # # print(res[i] + " : " + str(vidProcessed))


    # # #print(target_time_string[i])




# # # create an Empty DataFrame object
# # df = pd.DataFrame()
 

 
# # # append columns to an empty DataFrame
# # df['FileName'] = ['Anna', 'Pete', 'Tommy']
# # df['Label'] = [97, 600, 200]

# # print(df)

# # df2 = pd.DataFrame()
# # df2['FileName'] = ['Anna']
# # df2['Label'] = [97]

# # df = df.append(df2, ignore_index = True)

# # print(df)





# # df1 = pd.read_csv("faceDataLabels\\" + "faceLabels.txt")

# # #print(df1)

# # for i in range(0,df1.shape[0]):

    # # name, label = df1.iloc[i,0].split(" ")
    # # print(name)
    # # print(label)
    # # #print(df1.iloc[i,0])

    # # #print(text_df.iloc[i])