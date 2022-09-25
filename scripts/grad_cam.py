import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms ,models
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from yaml import load
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from PIL import *
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2

class Net(nn.Module):
    # def __init__(self, n_channel, n_out):
    #     super().__init__()
    # #<Network CNN 3 + FC 2>
    #     self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
    #     self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
    #     self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
    #     self.fc4 = nn.Linear(960, 512)
    #     self.fc5 = nn.Linear(512,n_out)
    #     self.relu = nn.ReLU(inplace=True)
    # #<Weight set>
    #     torch.nn.init.kaiming_normal_(self.conv1.weight)
    #     torch.nn.init.kaiming_normal_(self.conv2.weight)
    #     torch.nn.init.kaiming_normal_(self.conv3.weight)
    #     torch.nn.init.kaiming_normal_(self.fc4.weight)
    #     # torch.nn.init.kaiming_normal_(self.fc5.weight)
    #     #self.maxpool = nn.MaxPool2d(2,2)
    #     #self.batch = nn.BatchNorm2d(0.2)
    #     self.flatten = nn.Flatten()
    # #<CNN layer>
    #     self.cnn_layer = nn.Sequential(
    #         self.conv1,
    #         self.relu,
    #         self.conv2,
    #         self.relu,
    #         self.conv3,
    #         self.relu,
    #         #self.maxpool,
    #         self.flatten
    #     )
    # #<FC layer (output)>
    #     self.fc_layer = nn.Sequential(
    #         self.fc4,
    #         self.relu,
    #         self.fc5,
    #     )
    #
    # #<forward layer>
    # def forward(self,x):
    #     x1 = self.cnn_layer(x)
    #     x2 = self.fc_layer(x1)
    #     return x2

    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network MobileNetV2>
        self.v2 = models.mobilenet_v2(pretrained=True)
        self.v2.classifier[1]= nn.Linear(in_features=self.v2.last_channel, out_features = n_out)
    # <CNN layer>
        self.v2_layer = self.v2
    # <forward layer>
    def forward(self, x):
        x1 = self.v2_layer(x)
        return x1

    def grad_cam(self):
        device = torch.device("cpu")
        print(device)
        model = Net(3,1)
        # print(model)
        model.to(device)
        model.load_state_dict(torch.load('/home/kiyooka/catkin_ws/src/nav_cloning/data/model_with_dir_use_dl_output/mobilenetv2/model_gpu.pt' ), strict=False)
        model.eval()
        # model.state_dict()
        img = cv2.imread("/home/kiyooka/catkin_ws/src/nav_cloning/data/analysis/use_dl_output/1.jpg")
        # print(img.shape)
        img = resize(img, (48,64), mode='constant')
        x = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x.to(device)
        grad_cam = GradCAM(model=model, feature_layer=list(model.v2.classifier[0](x)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model_output = self.forward(x)
        target = model_output.argmax(1).item()
        grad_cam.backward_on_target(model_output, target)

        # Get feature gradient
        feature_grad = grad_cam.save_feature_grad.data.numpy()[0]
        # print(numpy()[0])
        # feature_grad = grad_cam.feature_grad.numpy()[0]
        # Get weights from gradient
        # weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient
        # Get features outputs
        # feature_map = grad_cam.feature_map.data.numpy()
        grad_cam.clear_hook()


class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.model.eval()
        self.feature_grad = None
        self.feature_map = None
        self.hooks = []

        # 最終層逆伝播時の勾配を記録する
    def save_feature_grad(module, in_grad, out_grad):
        self.feature_grad = out_grad[0]
        self.hooks.append(self.feature_layer.register_backward_hook(save_feature_grad))

    # 最終層の出力 Feature Map を記録する
    def save_feature_map(module, inp, outp):
        self.feature_map = outp[0]
        self.hooks.append(self.feature_layer.register_forward_hook(save_feature_map))

    def forward(self, x):
        return self.model(x)

    def backward_on_target(self, output, target):
        self.model.zero_grad()
        one_hot_output = torch.zeros([1, 1])
        one_hot_output[0][0] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

    def clear_hook(self):
        for hook in self.hooks:
            hook.remove()




if __name__ == '__main__':
        grad = Net(3,1)
        grad.grad_cam()
