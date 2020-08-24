import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import pandas as pd

# constants:
train_num_classes = 10
solution_num_classes = 9

# FC only:
class FC_1(torch.nn.Module):
    # model

    def __init__(self):
        super(FC_1, self).__init__()

        # Input channels = 10x9x9 (one hot vector of 0-9), output = 32x10x10
        self.conv1 = torch.nn.Conv2d(10, 32, kernel_size=3, stride=3, padding=0)
        # from 32x10x10 to 32x11x11
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # from 32x11x11 to 32x12x12
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 9 ** 3)
        self.bn1 = torch.nn.BatchNorm1d(9 ** 3)

        self.fc2 = torch.nn.Linear(9 ** 3, 9 ** 3)
        self.bn2 = torch.nn.BatchNorm1d(9 ** 3)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(9 ** 3, 9 ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 10*9*9)
        x = x.reshape(-1, 10 * 9 * 9)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).view(-1, 9, 9, 9)
        x = self.soft(x)
        return x #

# CNN:
class CNN_1(torch.nn.Module):

    def __init__(self):
        super(CNN_1, self).__init__()
        self.kenels_num = 128
        self.conv1_9 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=(1, 9), stride=1, padding=0)
        self.conv9_1 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=(9, 1), stride=1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=3, stride=3, padding=0)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(3 * 9 * self.kenels_num, 9 ** 4)
        self.bn1 = torch.nn.BatchNorm1d(9 ** 4)

        self.fc2 = torch.nn.Linear(9 ** 4, 9 ** 3)
        self.bn2 = torch.nn.BatchNorm1d(9 ** 3)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(9 ** 3, solution_num_classes ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y1 = self.conv1_9(x).reshape(-1, 9 * self.kenels_num)
        y2 = self.conv3_3(x).reshape(-1, 9 * self.kenels_num)
        x = self.conv9_1(x).reshape(-1, 9 * self.kenels_num)

        # x = torch.cat((y1.view(-1, 9*self.kenels_num), x.view(-1, 9*self.kenels_num)), dim=1)
        x = torch.cat((y1, y2, x), dim=1)
        # x = x.view(-1, 9*self.kenels_num)
        # Computes the activation of the first fully connected layer
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc3(x).view(-1, solution_num_classes, solution_num_classes, solution_num_classes)
        x = self.soft(x)
        return x


class ensamble(torch.nn.Module):
    def __init__(self, CNNnetDict, FCnetDict):
        super(ensamble, self).__init__()
        self.CNNnet = CNN_1()
        self.CNNnet.load_state_dict(CNNnetDict)
        self.FCnet = FC_1()
        self.FCnet.load_state_dict(FCnetDict)

    def forward(self, x):
        y1 = self.CNNnet(x)
        y2 = self.FCnet(x)
        return y1 + y2
