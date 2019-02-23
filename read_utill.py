import os
import csv
import sys
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
#from DataSet import Bucket

train_path = 'C:\\Users\\WeiLong\\PycharmProjects\\01-16\\training2017\\training2017'
save_path = 'C:\\Users\\liuNian\\PycharmProjects\\PaperTime'
test_path = 'C:\\Users\\WeiLong\\PycharmProjects\\01-16\\sample2017\\validation'
def label2num(element):
    if element == 'N':
        return 0
    elif element == 'O':
        return 1
    elif element == 'A':
        return 2
    elif element == '~':
        return 3

# def label2onehot_ex(element):
#     if element == '1':
#         return [1,0,0,0,0]
#     elif element == '2':
#         return [0,1,0,0,0]
#     elif element == '3':
#         return [0,0,1,0,0]
#     elif element == '4':
#         return [0,0,0,1,0]
#     elif element == '5':
#         return [0,0,0,0,1]

def normalization(array):
  # mean = np.mean(array)
  # std = np.std(array,axis=0)
  # newarray = (array - mean) / std
  # return np.reshape(newarray,[-1])
  return preprocessing.scale(array)


# def get_label(label_path):
#
#   def read_labels(label_path):
#     labels = []
#     with open(label_path) as file:
#       read = csv.reader(file)
#       for label_ in read:
#         labels.append(label_)
#     return labels
#
#   labels = read_labels(label_path)
#   row = len(labels)
#   label = []
#   for i in range(row):
#     label.append(label2onehot(labels[i][1]))
#   return label

def get_label_For_DataSet(path):
    label = {}
    with open(path) as file:
        reader = csv.reader(file)
        for i in reader:
            label[i[0]] = label2num(i[1])
    return label


def get_mod(Size, cover, origin_length):
    cycle = Size - cover
    length = 0
    while (1):
        if length == 0:
            left = origin_length // Size
            right = origin_length % Size
        else:
            left = origin_length // cycle
            right = origin_length % cycle
        length += 1
        if left == 1:
            if right == 0:
                return length
            else:
                if right < (cycle / 2):
                    return length
                else:
                    return length + 1
        else:
            if left == 0:
                return -1
            if length == 1:
                origin_length -= Size
            else:
                origin_length -= cycle

def padding_zero(data, right, size, cover):
    if right != 0:
        pad_length = size - cover - right
        data = np.lib.pad(data, (0, pad_length), 'constant', constant_values=(0, 0))
    return data

def preprocessTrainData(train_path):
    label = get_label_For_DataSet(os.path.join(train_path, 'REFERENCE.csv'))
    train_mat = os.listdir(train_path)
    train_data = []
    train_label = []
    for i in train_mat:
        if '.mat' in i:
            file = sio.loadmat(os.path.join(train_path, i))
            normalized_data = normalization(file['val'][0])
            train_data.append(normalized_data)
            #print(len(normalized_data))
            file_index = i[0:6]
            train_label.append(label[file_index])
    return train_data,train_label

def preprocessTestData(test_path):
    label = get_label_For_DataSet(os.path.join(test_path, 'REFERENCE.csv'))
    mat = os.listdir(test_path)
    test_data = []
    test_label = []
    for i in mat:
        if '.mat' in i:
            file = sio.loadmat(os.path.join(test_path, i))
            data = normalization(file['val'][0])
            test_data.append(data)
            file_index = i[0:6]
            test_label.append(label[file_index])
    return test_data,test_label

def readTrain(train_path,Size=300,cover=50):
    train_data, train_label = preprocessTrainData(train_path)
    train_length = []
    #########train_length#########
    for i in train_data:
        train_length.append(get_mod(Size,cover,len(i)))
    Max = Size + (np.max(train_length)-1)*(Size-cover)
    for i in range(len(train_data)):
        train_data[i] = np.lib.pad(train_data[i],(0,np.max((0,Max-len(train_data[i])))),'constant',constant_values=(0,0))
    new_train = []
    start = 0
    for index in range(len(train_data)):
        for i in range(np.max(train_length)):
            if i == 0:
                end = start + Size
                new_train.append(train_data[index][start:end])
            else:
                start = end - cover
                end = start + Size
                new_train.append(train_data[index][start:end])
    train = []
    assert len(train_data) == len(train_label)
    for i in range(len(train_data)):
        train.append((train_data[i],train_label[i],train_length[i]))
    return train

def readTest(test_path,Size=300,cover=50):
    test_data, test_label = preprocessTestData(test_path)
    test_length = []
    # #---Test---
    # for i in train_data:
    #     print(len(i))
    ######test_length########
    for i in test_data:
        test_length.append(get_mod(Size, cover, len(i)))
    Max = Size + (np.max(test_length)-1)*(Size-cover)
    for i in range(len(test_data)):
        test_data[i] = np.lib.pad(test_data[i],(0, np.max((0,Max-len(test_data[i])))), 'constant', constant_values=(0,0))
    # # ---Test---
    # for i in test_data:
    #     print(len(i))
    new_test = []
    start = 0
    for index in range(len(test_data)):
        for i in range(np.max(test_length)):
            if i == 0:
                end = start + Size
                new_test.append(test_data[index][start:end])
            else:
                start = end - cover
                end = start + Size
                new_test.append(test_data[index][start:end])
    test = []
    for i in range(len(test_data)):
        test.append((test_data[i],test_label[i],test_length[i]))
    # #---Test----
    # for i in train:
    #     last = Size + (i[2]-1)*(Size-cover)
    #     print(i[0][-last-30:-last],': ',i[2])
    # print(len(train))
    return test


#train = readTrain(train_path,Size=300,cover=0)
#test = readTest(test_path,Size=300,cover=0)
#for i  in train:
#    print(i[0].shape,' ',i[1],' ',i[2])
#print(len(train))