import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


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
#from torch.utils.tensorboard import SummaryWriter
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    #<Network CNN 3 + FC 2> 
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.fc6 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
    #<Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    #<CNN layer>   
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            #self.maxpool,
            self.flatten
        )
    #<FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            # self.fc5,
        )
        self.fc_out_layer = nn.Sequential(
            self.fc5,
        )
        self.fc_out_layer2 = nn.Sequential(
            self.fc6,
        )

    #<forward layer>
    def forward(self,x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        x3 = self.fc_out_layer(x2)
        x4 = self.fc_out_layer(x2)
        return x3, x4

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        #<tensor device choiece>
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),eps=1e-2,weight_decay=5e-4)
        #self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.datas = []
        self.target_angles = []
        self.criterion = nn.MSELoss()
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.first_flag =True
        torch.backends.cudnn.benchmark = True
        #self.writer = SummaryWriter(log_dir="/home/haru/nav_ws/src/nav_cloning/runs",comment="log_1")

    def make_dataset(self,img,target_angle, target_linear):
        if self.first_flag:
            self.x_cat = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat=self.x_cat.permute(0,3,1,2)
            self.t_cat = torch.tensor([target_angle],dtype=torch.float32,device=self.device).unsqueeze(0)
            self.t2_cat = torch.tensor([target_linear],dtype=torch.float32,device=self.device).unsqueeze(0)
            self.first_flag =False
        x = torch.tensor(img,dtype =torch.float32, device=self.device).unsqueeze(0)
        x=x.permute(0,3,1,2)
        t = torch.tensor([target_angle],dtype=torch.float32,device=self.device).unsqueeze(0)
        t2 = torch.tensor([target_linear],dtype=torch.float32,device=self.device).unsqueeze(0)
        self.x_cat =torch.cat([self.x_cat,x],dim=0)
        self.t_cat =torch.cat([self.t_cat,t],dim=0)
        self.t2_cat =torch.cat([self.t2_cat,t2],dim=0)
        
    #<make dataset>
        self.dataset = TensorDataset(self.x_cat,self.t_cat, self.t2_cat)

    def trains(self):
        self.net.train()
        train_dataset = DataLoader(self.dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'),shuffle=True)
        
    #<only cpu>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)
        
    #<split dataset and to device>
        for x_train, t_train, t2_train in train_dataset:
            x_train.to(self.device,non_blocking=True)
            t_train.to(self.device,non_blocking=True)
            t2_train.to(self.device,non_blocking=True)
            break

    #<learning>
        self.optimizer.zero_grad()
        y_train, y2_train = self.net(x_train)
        loss1 = self.criterion(y_train, t_train) 
        loss2 = self.criterion(y2_train, t2_train) 
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def act_and_trains(self, img,target_angle, target_linear):
        self.make_dataset(img,target_angle, target_linear)
        loss = self.trains()
        #<test>
        self.net.eval()
        x = torch.tensor(img,dtype =torch.float32, device=self.device).unsqueeze(0)
        x=x.permute(0,3,1,2)
        action_value_training, action2_value_training = self.net(x)
        return action_value_training[0][0].item(),action2_value_training[0][0].item(),  loss

    def act(self, img):
            self.net.eval()
        #<make img(x_test_ten),cmd(c_test)>
            # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
            x_test_ten = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
            x_test_ten = x_test_ten.permute(0,3,1,2)
            #print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        #<test phase>
            action_value_test, action2_value_test = self.net(x_test_ten)
            
            # print("act = " ,action_value_test.item())
            return action_value_test.item(), action2_value_test.item()

    def result(self):
            accuracy = self.accuracy
            return accuracy

    def save(self, save_path):
        #<model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')


    def load(self, load_path):
        #<model load>
        self.net.state_dict(torch.load(load_path))

if __name__ == '__main__':
        dl = deep_learning()
