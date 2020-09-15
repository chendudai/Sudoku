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
        self.name = "FC_1"

        # input 9x9x10 one-hot matrix
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 9 ** 3)
        self.bn1 = torch.nn.BatchNorm1d(9 ** 3)
        # self.drop1 = torch.nn.Dropout(p=0.4)

        self.fc2 = torch.nn.Linear(9 ** 3, 9 ** 3)
        self.bn2 = torch.nn.BatchNorm1d(9 ** 3)
        # self.drop2 = torch.nn.Dropout(p=0.4)


        self.fc3 = torch.nn.Linear(9 ** 3, 9 ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 10*9*9)
        x = x.reshape(-1, 10 * 9 * 9)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).view(-1, 9, 9, 9)
        x = self.soft(x)
        return x


# FC only:
class FC_shallow(torch.nn.Module):
    # model

    def __init__(self):
        super(FC_shallow, self).__init__()
        self.name = "FC_shallow"

        # input 9x9x10 one-hot matrix
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 2**10)
        self.bn1 = torch.nn.BatchNorm1d(2**10)
        self.drop1 = torch.nn.Dropout(p=0.4)

        self.fc3 = torch.nn.Linear(2**10, 9 ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 10*9*9)
        # x = x.reshape(-1, 10 * 9 * 9)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc3(x).view(-1, 9, 9, 9)
        x = self.soft(x)
        return x


# CNN:
class CNN_1(torch.nn.Module):

    def __init__(self):
        super(CNN_1, self).__init__()
        self.name = "CNN_1"
        self.kenels_num = 64
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
        self.fc3 = torch.nn.Linear(9 ** 4, solution_num_classes ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y1 = self.conv1_9(x).reshape(-1, 9 * self.kenels_num)
        y2 = self.conv3_3(x).reshape(-1, 9 * self.kenels_num)
        x = self.conv9_1(x).reshape(-1, 9 * self.kenels_num)

        # x = torch.cat((y1.view(-1, 9*self.kenels_num), x.view(-1, 9*self.kenels_num)), dim=1)
        x = torch.cat((y1, y2, x), dim=1)
        # x = x.view(-1, 9*self.kenels_num)
        # Computes the activation of the first fully connected layer
        x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc3(x).view(-1, solution_num_classes, solution_num_classes, solution_num_classes)
        x = self.soft(x)
        return x


# reference net:
class FC_ref(torch.nn.Module):
    # model

    def __init__(self):
        super(FC_ref, self).__init__()
        self.name = "FC_ref"

        # input 9x9x10 one-hot matrix
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 64)
        self.drop1 = torch.nn.Dropout(p=0.4)
        self.fc2 = torch.nn.Linear(64, 64)
        self.drop2 = torch.nn.Dropout(p=0.4)

        self.fc3 = torch.nn.Linear(64, 9 ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 10*9*9)
        # x = x.reshape(-1, 10 * 9 * 9)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x).view(-1, 9, 9, 9)
        x = self.soft(x)
        return x


class ensamble(torch.nn.Module):
    def __init__(self, CNNnetDict, FCnetDict):
        super(ensamble, self).__init__()
        self.name = "ensamble"
        self.CNNnet = CNN_1().load_state_dict(CNNnetDict)
        self.FCnet = FC_1().load_state_dict(FCnetDict)

    def forward(self, x):
        y1 = self.CNNnet(x)
        y2 = self.FCnet(x)
        return y1 + y2


class ensambleSameNet(torch.nn.Module):
    def __init__(self, dictsList, netType):
        super(ensambleSameNet, self).__init__()
        self.name = "ensambleSameNet"
        self.nets = []
        for dict in dictsList:
            self.nets.append(netType().load_state_dict(dict))

    def forward(self, x):
        y = np.zeros((len(x), 9, 9))
        for net in self.nets:
            y += net(x)
        return y
