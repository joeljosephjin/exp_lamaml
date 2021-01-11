import random
from random import shuffle

import numpy as np
# import ipdb
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import model.meta.learner as Learner
import model.meta.modelfactory as mf
from scipy.stats import pearsonr
import datetime

import learn2learn as l2l



# a subset of BaseNet containing the model, parameters, forward, backward, differentiation,etc..
# BaseNet comes from "lamaml_base.py" 
class Net(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
#         super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        super(Net, self).__init__()

        self.args = args
        nl, nh = args.n_layers, args.n_hiddens

        config = mf.ModelFactory.get_model(model_type = args.arch, sizes = [n_inputs] + [nh] * nl + [n_outputs],
                                                dataset = args.dataset, args=args)

        self.net = Learner.Learner(config, args)
        self.net = l2l.algorithms.MetaSGD(self.net, lr=0.001)

        # define the lr params
        self.net.define_task_lr_params(alpha_init = args.alpha_init)

        self.opt_wt = torch.optim.SGD(list(self.net.parameters()), lr=args.opt_wt)     
        self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=args.opt_lr) 

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
    def forward(self, x, t):
        # push x thru model and get y_pred
        output = self.net.forward(x)
        return output

    # get loss and y_pred normally
    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        # simply pushing the x forward thru the net
        # fast_weights doesn't seem to be getting used
        logits = self.net.forward(x, fast_weights)
        # get loss as usual
        loss_q = self.loss(logits.squeeze(1), y)
        # return the loss and the output y_pred
        return loss_q, logits

    # return an update of the fast weights
    def inner_update(self, x, model_clone, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        # same usual finding y_pred
#         logits = self.net.forward(x, fast_weights)
        logits = model_clone(x)
        # same usual finding loss without squeezing (i wonder why?)
#         loss = self.loss(logits, y)
        loss = self.loss(logits, y)

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        
        model_clone.adapt(loss)        
        
        return model_clone

    # idk??
    def observe(self, x, y, t):
        # initialize for training, make use of batch norms, dropouts,etc..
        self.net.train()
        
        opt = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # glances??
        for pass_itr in range(self.glances):
            model_clone = self.net.clone()
            # pass_itr is to be used in other funcs(of this class) ig
            self.pass_itr = pass_itr
            
            # Returns a random permutation of integers from 0 to n - 1
            perm = torch.randperm(x.size(0))
            # selecting a random data tuple (x, y)
            x, y = x[perm], y[perm]
            
            # so each it of this loop is an epoch
            self.epoch += 1
            # set {opt_lr, opt_wt, net, net.alpha_lr} (all 4) as zero_grads
            self.zero_grads()

            # current_task=??
            if t != self.current_task:
                # M=??
                self.M = self.M_new
                self.current_task = t

            # get batch_size from the shape of x
            batch_sz = x.shape[0]
            # will want to store batch loss in a list
            meta_losses = [0 for _ in range(batch_sz)] 

            # b_lisst <= {x,y,t} + sample(Memory)
            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            fast_weights = None
            
            # for each tuple in batch
            for i in range(batch_sz):
                # squeeze the tuples
                batch_x, batch_y = x[i].unsqueeze(0), y[i].unsqueeze(0)
                
                # do an inner update
                model_clone = self.inner_update(batch_x, model_clone, batch_y, t)

                # if real_epoch is zero, push the tuple to memory
                if(self.real_epoch == 0): self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                # get meta_loss and y_pred
#                 meta_loss, logits = self.meta_loss(bx, fast_weights, by, t)
                meta_loss = self.loss(model_clone(bx), by)
                # collect meta_losses into a list
                meta_losses[i] += meta_loss
    
            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            # get avg of the meta_losses
            meta_loss = sum(meta_losses)/len(meta_losses)

            # do bkwrd
            meta_loss.backward()
    
            opt.step()
                    
            # set zero_grad for net and alpha learning rate
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()
    
    def zero_grads(self):
        if self.args.learn_lr: self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()
        
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
        if x is not None: mxi, myi, mti = np.array(x), np.array(y), np.ones(x.shape[0], dtype=int)*t
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

        # b_lists <= add_to <= m_lists
        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        # b_lists <= torch-ize
        bxs = Variable(torch.from_numpy(np.array(bxs))).float() 
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # b_lists <= cuda-ize
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs,bys,bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)