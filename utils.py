import datetime
import glob
import json
import os
import random
# import ipdb
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import argparse


####################################METRICS.PY#############################################
def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(result_t, result_a, log_dir, fname=None):
    nt, changes = task_changes(result_t)
    fname = os.path.join(log_dir, fname)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[(torch.LongTensor(changes))]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        f.close()

    colors = cm.nipy_spectral(np.linspace(0, 1, len(result)))
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    data = np.array(result_a)
    for i in range(len(data[0])):
        plt.plot(range(data.shape[0]), data[:,i], label=str(i), color=colors[i], linewidth=2)
        
    plt.savefig(log_dir + '/' + 'task_wise_accuracy.png')

    stats = []
    stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
####################################METRICS.PY#############################################





def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot

def _check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.).item())

def compute_accuracy(ypred, ytrue, task_size=10):
    all_acc = {}

    all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(
                np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
        )[0]

        label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def get_date_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-2]

# log_dir(args, timestamp)
def log_dir(opt, timestamp=None):
    if timestamp is None:
        timestamp = get_date_time()

    rand_num = str(random.randint(1,1001))
    logdir = opt.log_dir + '/%s/%s-%s/%s' % (opt.model, opt.expt_name, timestamp, opt.seed)
    tfdir = opt.log_dir +  '/%s/%s-%s/%s/%s' % (opt.model, opt.expt_name, timestamp, opt.seed, "tfdir")

    mkdir(logdir)
    mkdir(tfdir)
    
    with open(logdir + '/training_parameters.json', 'w') as f:
        json.dump(vars(opt), f, indent=4)
    
    return logdir, tfdir


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def find_latest_checkpoint(folder_path):
    print('searching for checkpoint in : '+folder_path)
    files = sorted(glob.iglob(folder_path+'/*.pth'), key=os.path.getmtime, reverse=True)
    print('latest checkpoint is:')
    print(files[0])
    return files[0]


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    print("Set seed", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.enabled = False


def find_latest_checkpoint_name(folder_path):
    print('searching for checkpoint in : '+folder_path)
    files = glob.glob(folder_path+'/*.pth')
    min_num = 0
    filename = ''
    for i, filei in enumerate(files):
        ckpt_name = os.path.splitext(filei) 
        ckpt_num = int(ckpt_name.split('_')[-1])
        if(ckpt_num>min_num):
            min_num = ckpt_num
            filename = filei
    print('latest checkpoint is:')
    print(filename)
    return filename


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))


def log_sum_exp(input, dim=None, keepdim=False):
    """Numerically stable LogSumExp.

    Args:
        input (Tensor)
        dim (int): Dimension along with the sum is performed
        keepdim (bool): Whether to retain the last dimension on summing

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output


#########################################PARSER.PY###########################################
def get_parser():
    parser = argparse.ArgumentParser(description='Continual learning')
    parser.add_argument('--expt_name', type=str, default='test_lamaml',
                    help='name of the experiment')
    
    # model details
    parser.add_argument('--model', type=str, default='single',
                        help='algo to train')
    parser.add_argument('--arch', type=str, default='linear', 
                        help='arch to use for training', choices = ['linear', 'pc_cnn'])
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--xav_init', default=False , action='store_true',
                        help='Use xavier initialization')



    # optimizer parameters influencing all models
    parser.add_argument("--glances", default=1, type=int,
                        help="Number of times the model is allowed to train over a set of samples in the single pass setting") 
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all ' +
                        'experiments). Variable name is from GEM project.')
    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay.')
    parser.add_argument('--memories', type=int, default=5120, 
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (For baselines)')

    
    # experiment parameters
    parser.add_argument('--cuda', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=1000,
                        help='frequency of checking the validation accuracy, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--tf_dir', type=str, default='',
                        help='(not set by user)')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--loader', type=str, default='task_incremental_loader',
                        help='data loader to use')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', default=False, action='store_true',
                        help='present tasks in order')
    parser.add_argument('--classes_per_it', type=int, default=4,
                        help='number of classes in every batch')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='number of classes in every batch')
    parser.add_argument("--dataset", default="mnist_rotations", type=str,
                    help="Dataset to train and test on.")
    parser.add_argument("--workers", default=3, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-order", "--class_order", default="old", type=str,
                        help="define classes order of increment ",
                        choices = ["random", "chrono", "old", "super"])
    parser.add_argument("-inc", "--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument('--test_batch_size', type=int, default=100000 ,
                        help='batch size to use during testing.')


    # La-MAML parameters
    parser.add_argument('--opt_lr', type=float, default=1e-1,
                        help='learning rate for LRs')
    parser.add_argument('--opt_wt', type=float, default=1e-1,
                        help='learning rate for weights')
    parser.add_argument('--alpha_init', type=float, default=1e-3,
                        help='initialization for the LRs')
    parser.add_argument('--learn_lr', default=False, action='store_true',
                        help='model should update the LRs during learning')
    parser.add_argument('--sync_update', default=False , action='store_true',
                        help='the LRs and weights should be updated synchronously')

    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')
    parser.add_argument("--cifar_batches", default=3, type=int,
                        help="Number of batches in inner trajectory") 
    parser.add_argument('--use_old_task_memory', default=False, action='store_true', 
                        help='Use only old task samples for replay buffer data')    
    parser.add_argument('--second_order', default=False , action='store_true',
                        help='use second order MAML updates')


   # memory parameters for GEM | AGEM | ICARL 
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--steps_per_sample', default=1, type=int,
                        help='training steps per batch')


    # parameters specific to MER 
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma learning rate parameter')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta learning rate parameter')
    parser.add_argument('--s', type=float, default=1,
                        help='current example learning rate multiplier (s)')
    parser.add_argument('--batches_per_example', type=float, default=1,
                        help='the number of batch per incoming example')


    # parameters specific to Meta-BGD
    parser.add_argument('--bgd_optimizer', type=str, default="bgd", choices=["adam", "adagrad", "bgd", "sgd"],
                    help='Optimizer.')
    parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                        help='Optimizer parameters')

    parser.add_argument('--train_mc_iters', default=5, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--std_init', default=5e-2, type=float,
                        help='STD init value (default 5e-2)')
    parser.add_argument('--mean_eta', default=1, type=float,
                        help='Eta for mean step (default 1)')
    parser.add_argument('--fisher_gamma', default=0.95, type=float,
                        help='')

    return parser
#########################################PARSER.PY###########################################
