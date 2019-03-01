import torch
import numpy as np
from torchvision import transforms, utils
from read_utils import readTrain, readTest
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):

    def __init__(self, train_path, test_path, size=300, cover=0, train=False, test=False):
        self.isTrain = train
        self.isTest = test
        if self.isTrain:
            self.train = readTrain(train_path, size, cover)
        if self.isTest:
            self.test = readTest(test_path, size, cover)
    def __len__(self):
        if self.isTrain:
            return len(self.train)
        if self.isTest:
            return len(self.test)

    def __getitem__(self, index):
        if self.isTrain:
            return self.train[index]
        if self.isTest:
            return self.test[index]
        raise Exception("undefined train or test in params of ECGDataset.__init__")

train_path = 'C:\\Users\\WeiLong\\PycharmProjects\\01-16\\training2017\\training2017'
test_path = 'C:\\Users\\WeiLong\\PycharmProjects\\01-16\\sample2017\\validation'

#train_loader = torch.utils.data.DataLoader(
#    ECGDataset(train_path,test_path,train=True),
#    batch_size=1,
#    shuffle=True,
#    drop_last=True
#)
#test_loader = torch.utils.data.DataLoader(
#    ECGDataset(train_path,test_path,test=True),
#    batch_size=100,
#    shuffle=True,
#    drop_last=True
#)
#for batch_idx, data in enumerate(train_loader):
#    print(data[0][0].shape)