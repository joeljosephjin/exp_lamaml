import importlib
import datetime
import argparse
import time
import os

import torch
from torch.autograd import Variable
import utils
from utils import get_parser
from utils import confusion_matrix
import wandb

import lamaml as Model

import dataloader as Loader


# returns lists of avg loss
def eval_tasks(model, tasks, args):
    # prep for eval
    model.eval()
    result = []
    # for each task
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

            # cuda-ize xb if necessary
            if args.cuda: xb = xb.cuda()
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            # adding the loss each time to rt
            rt += (pb == yb).float().sum()
        # average loss of each task added to result list
        result.append(rt / x.size(0))

    return result

# for lamaml and everything except iid
def life_experience(model, inc_loader, args):
    wandb.init(project="exp_lamaml", entity="joeljosephjin")

    result_val_a, result_test_a = [], []
    result_val_t, result_test_t = [], []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    
    evaluator = eval_tasks

    # for no. of tasks
    # print(inc_loader.n_tasks)
    for task_i in range(inc_loader.n_tasks):
        # initialize new task
        task_info, train_loader, _, _ = inc_loader.new_task()
        # for no. of epochs
        for ep in range(args.n_epochs):

            model.real_epoch = ep

            prog_bar = train_loader
            # for each data tuple {x,y}
            for (i, (x, y)) in enumerate(prog_bar):

                if((i % args.log_every) == 0):
                    # get accuracy by evaluating on validation data ??
                    result_val_a.append(evaluator(model, val_tasks, args))
                    result_val_t.append(task_info["task"])

                # v_x/y <= x/y
                v_x, v_y = x, y
                # linearize the x i.e. image => list
                if args.arch == 'linear': v_x = x.view(x.size(0), -1)
                # cuda-ize the x and y
                if args.cuda: v_x, v_y = v_x.cuda(), v_y.cuda()

                model.train()

                loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])

                wandb.log({"Task": task_info["task"], "Epoch": ep+1/args.n_epochs, "Iter": i%(1000*args.n_epochs),
                 "Loss": round(loss, 3),
                 "Total Acc": round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5),
                  "Curr Task Acc": round(result_val_a[-1][task_info["task"]].item(), 5)})

        result_val_a.append(evaluator(model, val_tasks, args))
        result_val_t.append(task_info["task"])

    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent

def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')

    wandb.log({"result_val_t":result_val_t, "result_val_a":result_val_a, "result_test_t":result_test_t, "result_test_a":result_test_a})
    wandb.save(fname+'.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt')
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])

    wandb.save(fname+'.txt')

    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_val_t, result_val_a, model.state_dict(),
                val_stats, one_liner, args), fname + '.pt')
    return val_stats, test_stats

def main():
    # loads a lot of default parser values from the 'parser' file
    parser = get_parser()

    # get args from parser as an object
    args = parser.parse_args()

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
    if args.cuda:
        try: model.net.cuda()            
        except: pass 

    # for all the CL baselines
    result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
        model, loader, args)

    # save results in files or print on terminal
    save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)


if __name__ == "__main__":
    main()
