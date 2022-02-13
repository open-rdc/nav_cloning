import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser

# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000

class Net(chainer.Chain):
    def __init__(self, n_channel=3, n_action=1):
        initializer = chainer.initializers.HeNormal()
        super(Net, self).__init__(
            conv1=L.Convolution2D(n_channel, 32, ksize=8, stride=4, nobias=False, initialW=initializer),
            conv2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False, initialW=initializer),
            conv3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, initialW=initializer),
            fc4=L.Linear(960, 512, initialW=initializer),
            fc5=L.Linear(512, 256, initialW=initializer),
            fc6=L.Linear(260, 260, initialW=initializer),#img=256 cmd=[0,0,0,0]=4 256+4=260
            fc7=L.Linear(260, n_action, initialW=np.zeros((n_action, 260), dtype=np.int32))
            )
    def __call__(self, x, c, test=False):
        s = chainer.Variable(x)
        h1 = F.relu(self.conv1(s))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        s2 = chainer.Variable(c)
        h6 = F.relu(self.fc6(F.concat((h5,s2),axis=1)))
        h = self.fc7(h6)
        return h

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        self.net = Net(n_channel, n_action)
        self.optimizer = chainer.optimizers.Adam(eps=1e-2)
        self.optimizer.setup(self.net)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
        self.n_action = n_action
        self.phi = lambda x: x.astype(np.float32, copy=False)
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.data = []
        self.cmd =[]
        self.target_angles = []

    def act_and_trains(self, imgobj, cmd_dir, target_angle):
            x = [self.phi(s) for s in [imgobj]]
            c = np.array([cmd_dir], np.float32)
            t = np.array([target_angle], np.float32)
            self.data.append(x[0])
            self.cmd.append(c[0])
            self.target_angles.append(t[0])
            if len(self.data) > MAX_DATA:
                del self.data[0]
                del self.cmd[0]
                del self.target_angles[0]
            dataset = TupleDataset(self.data, self.cmd, self.target_angles)
            train_iter = SerialIterator(dataset, batch_size = BATCH_SIZE, repeat=True, shuffle=True)
            train_batch  = train_iter.next()
            x_train, c_train, t_train = chainer.dataset.concat_examples(train_batch, -1)

            y_train = self.net(x_train, c_train)
            loss_train = F.mean_squared_error(y_train, Variable(t_train.reshape(BATCH_SIZE, 1)))

            self.loss_list.append(loss_train.array)

            self.net.cleargrads()
            loss_train.backward()
            self.optimizer.update()
            
            self.count += 1

            self.results_train['loss'] .append(loss_train.array)
            x_test = chainer.dataset.concat_examples(x, -1)
            c_test = chainer.dataset.concat_examples(c, -1)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                action_value = self.net(x_test, c_test)
            return action_value.data[0][0], loss_train.array

    def act(self, imgobj, cmd_dir):
            x = [self.phi(s) for s in [imgobj]]
            c = np.array([cmd_dir],np.float32)
            x_test = chainer.dataset.concat_examples(x, -1)
            c_test = chainer.dataset.concat_examples(c,-1)

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                action_value = self.net(x_test, c_test)
            return action_value.data[0][0]

    def result(self):
            accuracy = self.accuracy
            return accuracy

    def save(self, save_path):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        chainer.serializers.save_npz(path + '/model.net' , self.net)

    def load(self, load_path):
        chainer.serializers.load_npz(load_path , self.net)

if __name__ == '__main__':
        dl = deep_learning()
