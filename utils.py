import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


class Results:
    def __init__(self, seed_num, model_num, args):
        super(Results, self).__init__()

        self.seed_num = seed_num
        self.model_num = model_num
        self.dataset = args.dataset
        self.model = args.model
        self.auc, self.f1, self.acc, self.parity, self.equality = np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num))

    def report_results(self):
        for i in range(self.model_num):
            print(f"============" + f"{self.dataset}" + "+" + f"{self.model}" + "============")
            print(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}")
            print(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}')
            print(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}')
            print(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}')
            print(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}')
            print("=================END=================")

    def save_results(self, args):
        for i in range(self.model_num):
            with open(f"{args.dataset}_tao{args.tem}_model{i}.txt", 'a') as f:
                f.write(f"τ={args.tem}, γ={args.alpha}, lr_w={args.lr_w}\n")
                f.write(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}\n")
                f.write(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}\n')
                f.write(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}\n')
                f.write(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}\n')
                f.write(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}\n')
            f.close()


class FeatCov(_Loss):
    def __init__(self):
        super(FeatCov, self).__init__()

    def forward(self, features, sens):
        cov = 0
        for k in range(features.shape[1]):
            cov += torch.abs(torch.mean((sens - torch.mean(sens)) * (features[:, k] - torch.mean(features[:, k]))))
        return cov


class DistCor(_Loss):
    def __init__(self):
        super(DistCor, self).__init__()

    def Distance_Correlation(self, channel1, channel2):
        assert channel1.shape[1] == channel2.shape[1], "The dim of two tensors need to be equal"
        dim_num = channel1.shape[1]
        correlation_r = 0
        for i in range(dim_num):
            latent = channel1[:, i]
            latent = latent.unsqueeze(1)
            control = channel2[:, i]
            control = control.unsqueeze(1)

            matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
            matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

            matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
            matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

            Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

            correlation_r += Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, channel1, channel2):
        dc_loss = self.Distance_Correlation(channel1, channel2)
        return dc_loss