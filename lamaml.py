import random
from random import shuffle

import numpy as np
# import ipdb
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy.stats import pearsonr
import datetime

import learn2learn as l2l



# a subset of BaseNet containing the model, parameters, forward, backward, differentiation,etc..
# BaseNet comes from "lamaml_base.py" 
class Net(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()

        self.args = args
        nl, nh = args.n_layers, args.n_hiddens

        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
            )
        
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
                
        self.net.apply(init_weights)

        self.net = l2l.algorithms.MetaSGD(self.net, lr=0.001)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.epoch = 0
        # allocate buffer
        self.M = []        
        self.M_new = []
        self.age = 0

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances
        self.pass_itr = 0
        self.real_epoch = 0

        self.current_task = 0
        self.memories = args.memories
        self.batchSize = int(args.replay_batch_size)

        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

        self.n_outputs = n_outputs

        self.nc_per_task = n_outputs

    # trivial forward
    def forward(self, x):
        # push x thru model and get y_pred
        output = self.net.forward(x)
        return output
        
    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()              
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]

    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch of old + new data, 
        where old data is sampled from the memory buffer
        """
        # numpy-ize the data{x,y,t}
        if x is not None: mxi, myi, mti = np.array(x.cpu()), np.array(y.cpu()), np.ones(x.shape[0], dtype=int)*t
        # if no data, then create empty numpy array
        else: mxi, myi, mti = np.empty( shape=(0, 0) ), np.empty( shape=(0, 0) ), np.empty( shape=(0, 0) )

        # might store the data into lists
        bxs, bys, bts = [], [], []

        # use from old memory or new memory
        if self.args.use_old_task_memory and t>0: MEM = self.M
        else: MEM = self.M_new

        # use self.batch_size if necessary
        batch_size = self.batchSize if batch_size is None else batch_size

        # if there is anything in memory
        if len(MEM) > 0:
            # order = {0,1,2...,len(MEM)-1}
            order = [i for i in range(len(MEM))]
            # run the loop until minm MEM or batch_siz
            osize = min(batch_size,len(MEM))
            for j in range(osize):
                # randomly shuffle the order list
                shuffle(order)
                # get random tuples of {x,y,t}
                x,y,t = MEM[order[j]]
                
                # numpy-ize the data tuple
                xi, yi, ti = np.array(x), np.array(y), np.array(t)

                # store the data tuple into the lists
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        bxs.extend(mxi)
        bys.extend(myi)
        bts.extend(mti)

        # b_lists <= torch-ize
        bxs = torch.from_numpy(np.array(bxs)).float().to(self.args.device)
        bys = torch.from_numpy(np.array(bys)).long().view(-1).to(self.args.device)
        bts = torch.from_numpy(np.array(bts)).long().view(-1).to(self.args.device)
        
        return bxs,bys,bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)