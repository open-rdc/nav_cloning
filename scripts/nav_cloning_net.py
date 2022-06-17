from pickletools import optimize
from platform import release
from pyexpat import model
import re
from turtle import forward
from typing_extensions import Self
from chainer import Optimizer
from gpg import Data
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser
from paramiko import Channel

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d()
        self.batch = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.flatten
        )
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5
        )

    def forward(self,x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2,weight_decay=5e-4)
        #self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.img_data = []
        self.dir_list =[]
        self.target_angles = []
        self.criterion = nn.MSELoss()
        self.transform=transforms.Compose([transforms.ToTensor()])

    def act_and_trains(self, imgobj, target_angle):
            self.net.train()
            self.img_data.append(imgobj)
            self.target_angles.append([target_angle])
            x = torch.tensor(self.img_data,dtype =torch.float32, device=self.device)
            # (Batch,Channel,H,W) -> (Batch ,Channel, H,W)
            x= x.permute(0,3,1,2)
            t = torch.tensor(self.target_angles,dtype=torch.float32,device=self.device)
            # self.img_data.append(x[0])
            # self.target_angles.append(t[0])
            # if len(self.img_data) > MAX_DATA:
            #     del self.img_data[0]
            #     del self.target_angles[0]

            dataset = TensorDataset(x,t)
            train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(device=self.device),shuffle=True)#train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(device=self.device),shuffle=True)
            
            #only cpu
            # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
            
            #split dataset
            for x_train,t_train in train_dataset:
                x_train.to(self.device)
                t_train.to(self.device)
                break
            
            #learning
            y_train = self.net(x_train)
            loss = self.criterion(y_train, t_train) 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad
            self.count += 1

            #test
            action_value_training = self.net(x)
            #print("action=" ,action_value_training[0][0].item() ,"loss=" ,loss.item())
            return action_value_training[0][0].item(), loss.item()

    def act(self, imgobj):
            self.net.eval()
            x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
            #print(x_test.shape,x_test.device,c_test.shape,c_test.device)
            action_value_test = self.net(x_test_ten)
            #print("act = " ,action_value_test.item())
            return action_value_test.item()

    def result(self):
            accuracy = self.accuracy
            return accuracy

    def save(self, save_path):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net, path + '/model.net')


    def load(self, load_path):
        self.net = torch.load(load_path)

if __name__ == '__main__':
        dl = deep_learning()
