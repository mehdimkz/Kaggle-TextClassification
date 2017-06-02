import sys
import numpy
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import itertools
from itertools import izip_longest
from itertools import islice
from io import BytesIO
import gzip
import os
import time
import gc
import csv
from sklearn.externals import joblib

def preprocessing(path, filename, test_flag=False): #Clean the orginal training and testing file to be compatible with Libsvm data format
    df1=pd.read_csv(path, sep='\t')
    if test_flag:
        file = open(filename, 'w')
        for i, r in df1.iterrows():
            r[0] = r[0].replace(',', ' ')
            str1 = (''.join(r[0].split(' ', 1)))
            str1 = str1.replace("0", "", 1)
            file.write(str1)
            file.write('\n')
        file.close()
    else:
        df1.columns = ['a']
        df1['a'] = df1['a'].map(lambda x: x.rstrip(','))
        df1['a'] = df1['a'].str.replace(r', ',',')
        file = open(filename, 'w')
        for i, r in df1.iterrows():
            file.write(','.join(r[0].split(',')))
            file.write('\n')
        file.close()

def preprocess2(filename,testfile): # Seprate lables of data samples
    qbfile = open(filename, "r")
    file = open(testfile, 'w')
    for aline in qbfile:
        values = aline.split()  # split each line based on the spaces
        label_string = values[0].split(',')  # split the first section of the text which includes multilabels to a list
        new_values = values[1:values.__len__()]  # extract the features of each row as alist
        new_values = ' '.join(new_values)  # Concatenate items in the list as string with a space seprator
        while label_string:
            # print test.pop()+' '+ new_values
            file.write(label_string.pop() + ' ' + new_values)
            file.write('\n')
    qbfile.close()
    file.close()


def preprocess3(fileoutput,lines_per_file,prefix):   #divide the text file to different chunk sizes

    smallfile = None
    with open(fileoutput) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = prefix.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def train(train_path, filename,lines_per_file,fileoutput,numtraining_models):
    preprocessing(train_path, filename)
    preprocess2(filename,fileoutput)
    preprocess3(fileoutput, lines_per_file, 'small_train_file_{}.txt')
    X_train, Y_train = load_svmlight_file(fileoutput, multilabel=False)
    global n_features
    n_features = X_train.shape[1] #Calculate total number of data features
    for k in range(1, numtraining_models):   #Create and save training models
        trainfile = 'small_train_file_' + str((k) * lines_per_file) + '.txt'
        X_train, Y_train = load_svmlight_file(trainfile, n_features=n_features, multilabel=False)
        print trainfile
        clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, max_features='auto', min_samples_split=2,
                                   random_state=0, n_jobs=-1)
        start = time.time()
        train_model = clf.fit(X_train, Y_train)
        train_save = 'trained_model' + str(k) + '.pkl'
        joblib.dump(train_model, train_save, compress=9)
        end = time.time()
        print(end - start)
        gc.collect()
     #return [n_features]


def test(test_path, filename,numtraining_models,lines_per_file):
  data = []
  preprocessing(test_path, filename,test_flag=True)
  preprocess3(filename, lines_per_file, 'small_test_file_{}.txt') #break test file to smaller sizes,100000 is the numebr of test samples in each file
  print n_features
  file = open(test_path)
  numline = len(file.readlines())
  print (numline)
  numtestfiles= numline/lines_per_file
  id=1
  for j in range(1, numtestfiles+1):
    test_file = 'small_test_file_' + str(j * lines_per_file) + '.txt'
    X_test, Y_test = load_svmlight_file(test_file,n_features=n_features,multilabel=False)
    for k in range(1, numtraining_models):
        train_file = 'trained_model' + str(k) + '.pkl'
        train_model = joblib.load(train_file)
        if k == 1:
            Prediction = train_model.predict(X_test)
        else:
            Prediction = np.vstack((train_model.predict(X_test), Prediction))
        print k

    # Find the frequency of each predicted label for each test sample
    Pre = Prediction.transpose()
    if numtestfiles == 1:
        f = open('Predictions_.csv', 'w')
        f.writelines("id,labels\n")

    else:
        f = open('Predictions_.csv', 'a')

    for i in range(0, Pre.shape[0]):
        unique, counts = np.unique(Pre[i], return_counts=True)
        Indx_highest = counts.argsort()[-3:][::-1]  # return index of top 2 values
        LabelsString = ''
        for j in range(0, 3):
            if counts[Indx_highest[j]] > 1:
                unique[Indx_highest[j]]
                LabelsString = LabelsString + str(unique[Indx_highest[j]]) + ', '
        f.write(str(id) + ',' + LabelsString + '\n')
        id = id + 1
    f.close()







def main():

     train('D:\Kaggle\\train-remapped\\train-remapped.csv', 'D:\Kaggle\\train.txt',12000,'D:\Kaggle\\train_main.txt',100)
     test('D:\Kaggle\\test-remapped\\test-remapped.csv', 'test-main.txt',100,100000)


main()
