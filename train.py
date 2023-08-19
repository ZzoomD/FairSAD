#%%
# import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


from load_data import *
from fairsad import *
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--channels", type=int, default=4, help="number of channels")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf',
                                                                     'RobustGCN', 'mlpgcn', 'gcnori', 'disengnn',
                                                                     'disengcn', 'pcagcn', 'adagcn', 'adagcn_new'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--alpha', type=float, default=0.25, help='weight coefficient for disentanglement')
    parser.add_argument('--beta', type=float, default=0.25, help='weight coefficient for channel masker')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--model_type', type=str, default='gnn', choices=['gnn', 'mlp', 'other'])
    parser.add_argument('--weight_path', type=str, default='./Weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    fair_dataset = FairDataset(args.dataset, args.device)
    fair_dataset.load_data()

    num_class = 1
    args.nfeat = fair_dataset.features.shape[1]
    args.nnode = fair_dataset.features.shape[0]
    args.nclass = num_class

    """
    Build model
    """
    fairsad_new = FairSAD(args)
    
    """
    Train model (Teacher model and Student model)
    """
    weight_path = f'./{args.model}_vanilla.pt'
    auc_roc_test, f1_s, acc, parity, equality = fairsad_new.train_fit(None, None, fair_dataset, args.epochs,
                                                                      model_type=args.model_type,
                                                                      weight_path=weight_path, alpha=args.alpha,
                                                                      beta=args.beta, pbar=args.pbar)

    return auc_roc_test, f1_s, acc, parity, equality


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num, args)

    for seed in range(args.seed_num):
        # set seeds
        args.pbar = tqdm(total=args.epochs, desc=f"Seed {seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        set_seed(seed)

        # running train
        results.auc[seed, :], results.f1[seed, :], results.acc[seed, :], results.parity[seed, :], \
        results.equality[seed, :] = run(args)

    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)
