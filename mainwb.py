import importlib
import datetime
import argparse
import time
import os

import torch
from torch.autograd import Variable
import utils
from utils import get_parser, confusion_matrix, save_results
import wandb

import lamaml as Model

import dataloader as Loader


def eval_tasks(model, tasks, args):
    model.eval()
    result = []
    
    for i, task in enumerate(tasks):

        t = i
        x, y = task[1], task[2]
        rt = 0
        
        eval_bs = x.size(0)

        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)

            if b_from == b_to: 
                xb, yb = x[b_from].view(1, -1), torch.LongTensor([y[b_to]]).view(1, -1)
            else: 
                xb, yb = x[b_from:b_to], y[b_from:b_to]

            xb = xb.to(args.device)
            
            _, pb = torch.max(model(xb).data.cpu(), 1, keepdim=False)
            # adding the accuracy each time to rt
            rt += (pb == yb).float().sum()
        # average accuracy on all tasks added to result list
        result.append(rt / x.size(0))

    return result


def life_experience(model, inc_loader, args):
    wandb.init(project="exp_lamaml", entity="joeljosephjin")

    result_val_a, result_test_a = [], []
    result_val_t, result_test_t = [], []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    
    evaluator = eval_tasks

    for task_i in range(inc_loader.n_tasks):
        
        task_info, train_loader, _, _ = inc_loader.new_task()
        
        for ep in range(args.n_epochs):

            model.real_epoch = ep

            # print(len(train_loader)) # 100
            
            for (i, (x, y)) in enumerate(train_loader):

                if((i % args.log_every) == 0):
                    result_val_a.append(evaluator(model, val_tasks, args))
                    result_val_t.append(task_info["task"])

                x, y = x.view(x.size(0), -1).to(args.device), y.to(args.device)

                model.train()

                #OBSERVE
                t = task_info["task"]
                
                for pass_itr in range(model.glances):
                    model_clone = model.net.clone()
                    model.pass_itr = pass_itr

                    # shuffle x,y randomly
                    perm = torch.randperm(x.size(0)); x, y = x[perm], y[perm]

                    model.epoch += 1

                    if t != model.current_task:
                        model.M = model.M_new
                        model.current_task = t

                    batch_sz = x.shape[0]
                    # print('batch sz', batch_sz) # 10
                    
                    meta_losses = [0 for _ in range(batch_sz)] 

                    # b_list <= {x,y,t} + sample(Memory)
                    bx, by, bt = model.getBatch(x, y, t)

                    for i in range(batch_sz):
                        
                        x_i, y_i = x[i].unsqueeze(0), y[i].unsqueeze(0) # x_i.shape = (1, 784)
                        
                        # print('batch shape', batch_x.shape) # [1, 784]

                        # do an inner update
                        loss = model.loss(model_clone(x_i), y_i)
                        model_clone.adapt(loss)

                        # if real_epoch is zero, push the tuple to memory
                        if(model.real_epoch == 0): model.push_to_mem(x_i, y_i, torch.tensor(t))

                        # get meta_loss
                        meta_loss = model.loss(model_clone(bx), by)
                        
                        # collect meta_losses into a list
                        meta_losses[i] += meta_loss

                    # get avg of the meta_losses
                    meta_loss = sum(meta_losses)/len(meta_losses)

                    # do bkwrd
                    model.net.zero_grad()
                    meta_loss.backward()
                    model.opt.step()

                wandb.log({"Task": task_info["task"], "Epoch": ep+1/args.n_epochs, "Iter": i%(1000*args.n_epochs),
                 "Loss": round(meta_loss.item(), 3),
                 "Total Acc": round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5),
                  "Curr Task Acc": round(result_val_a[-1][task_info["task"]].item(), 5)})

        result_val_a.append(evaluator(model, val_tasks, args))
        result_val_t.append(task_info["task"])

    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent


def main():
    # loads a lot of default parser values from the 'parser' file
    parser = get_parser()

    # get args from parser as an object
    args = parser.parse_args()
    args.device = 'cuda' if args.cuda else 'cpu'

    # initialize seeds
    utils.init_seed(args.seed)

    # print('loader stuff', args)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    # print('loader stuff after after', args)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    # setup logging
    # logging is from 'misc_utils.py' from 'utils' folder
    timestamp = utils.get_date_time() # this line is redundant bcz log_dir already takes care of it
    args.log_dir, args.tf_dir = utils.log_dir(args, timestamp) # stores args into "training_parameters.json"

    # create the model neural net
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    
    # make model cuda-ized if possible
    model.net.to(args.device)            

    # for all the CL baselines
    result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(model, loader, args)

    # save results in files or print on terminal
    save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)


if __name__ == "__main__":
    main()
