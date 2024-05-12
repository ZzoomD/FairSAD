import os
# import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import *
from torch_geometric.utils import dropout_adj, convert
from torch_sparse import SparseTensor
import torch.nn as nn


class FairDataset:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    @staticmethod
    def adj2edge_index(adj, fea_num):
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        edge_index_spar = SparseTensor.from_edge_index(edge_index, sparse_sizes=(fea_num, fea_num), )
        return edge_index, edge_index_spar

    def load_data(self, sens_attr=None, label_number=None, split_ratio=None, val_idx=True, sens_number=None):
        """
            Load data
        """
        # Load credit_scoring dataset
        if self.dataset == 'credit':
            dataset = 'credit'
            sens_attr = "Age" if sens_attr is None else sens_attr   # column number after feature process is 1
            sens_idx = 1   # column number of the sensitive attribute. Note that different sensitive attributes have different sens_idx
            predict_attr = 'NoDefaultNextMonth'
            label_number = 6000 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            path_credit = "./datasets/credit"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr,
                                                                                    predict_attr, path=path_credit,
                                                                                    label_number=label_number,
                                                                                    split_ratio=split_ratio,
                                                                                    val_idx=val_idx
                                                                                    )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features

        # Load german dataset
        elif self.dataset == 'german':
            dataset = 'german'
            sens_attr = "Gender"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "GoodCustomer"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            path_german = "./datasets/german"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                    predict_attr, path=path_german,
                                                                                    label_number=label_number,
                                                                                    split_ratio=split_ratio,
                                                                                    val_idx=val_idx
                                                                                    )
        # Load bail dataset
        elif self.dataset == 'bail':
            dataset = 'bail'
            sens_attr = "WHITE"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "RECID"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            path_bail = "./datasets/bail"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr,
                                                                                  predict_attr, path=path_bail,
                                                                                  label_number=label_number,
                                                                                  split_ratio=split_ratio,
                                                                                  val_idx=val_idx
                                                                                  )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
        # load pokec dataset
        elif self.dataset == 'pokec_z':
            dataset = 'region_job'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 4000 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./datasets/pokec/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
        elif self.dataset == 'pokec_n':
            dataset = 'region_job_2'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 3500 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./datasets/pokec/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
        elif self.dataset == 'nba':
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_idx = 35
            seed = 20
            path = "./datasets/NBA/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
        else:
            print('Invalid dataset name!!')
            exit(0)

        edge_index, edge_index_spar = FairDataset.adj2edge_index(adj=adj, fea_num=features.shape[0])
        edge_index_spar = edge_index_spar.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        sens = sens.to(self.device)
        self.label_number, self.sens_number = label_number, sens_number
        self.edge_index, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = \
            edge_index_spar, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(dataset, sens_attr, predict_attr, path="./datasets/pokec/", label_number=1000, sens_number=500,
               seed=19, split_ratio=None, val_idx=True):
    """Load data"""
    # print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    # delete edge connecting nodes with negative label
    # label_del_idx = np.where(labels < 0)[0]
    # del_target = np.isin(edges[:, 0], label_del_idx)
    # del_source = np.isin(edges[:, 1], label_del_idx)
    # del_edge = del_target + del_source
    # edges_new = edges[~del_edge]
    # adj = sp.coo_matrix((np.ones(edges_new.shape[0]), (edges_new[:, 0], edges_new[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    # seed = 20
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    # print("label_num", label_idx.shape)
    # print("label=0", np.where(labels == 0)[0].shape)
    random.shuffle(label_idx)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = label_idx[:min(int(split_ratio[0] * len(label_idx)), label_number)]
    if val_idx:
        idx_val = label_idx[int(split_ratio[0] * len(label_idx)):int((split_ratio[0] + split_ratio[1]) * len(label_idx))]
        idx_test = label_idx[int((split_ratio[0] + split_ratio[1]) * len(label_idx)):]
    else:
        idx_test = label_idx[label_number:]
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # random.shuffle(sens_idx)
    # a = torch.unique(labels)
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./datasets/credit/",
                label_number=1000, split_ratio=None, val_idx=True):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    #    # Normalize MaxBillAmountOverLast6Months
    #    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
    #
    #    # Normalize MaxPaymentAmountOverLast6Months
    #    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
    #
    #    # Normalize MostRecentBillAmount
    #    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
    #
    #    # Normalize MostRecentPaymentAmount
    #    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
    #
    #    # Normalize TotalMonthsOverdue
    #    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = np.append(label_idx_0[:min(int(split_ratio[0] * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(split_ratio[0] * len(label_idx_1)), label_number // 2)])
    if val_idx:
        idx_val = np.append(label_idx_0[int(split_ratio[0] * len(label_idx_0)):int(
            (split_ratio[0] + split_ratio[1]) * len(label_idx_0))],
                            label_idx_1[int(split_ratio[0] * len(label_idx_1)):int(
                                (split_ratio[0] + split_ratio[1]) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((split_ratio[0] + split_ratio[1]) * len(label_idx_0)):],
                             label_idx_1[int((split_ratio[0] + split_ratio[1]) * len(label_idx_1)):])
    else:
        idx_test = np.append(label_idx_0[min(int(split_ratio[0] * len(label_idx_0)), label_number // 2):],
                             label_idx_1[min(int(split_ratio[0] * len(label_idx_1)), label_number // 2):])
        idx_val = idx_test


    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./datasets/bail/", label_number=1000,
              split_ratio=None, val_idx=True):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # header.remove(sens_attr)

    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = np.append(label_idx_0[:min(int(split_ratio[0] * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(split_ratio[0] * len(label_idx_1)), label_number // 2)])

    if val_idx:
        idx_val = np.append(label_idx_0[int(split_ratio[0] * len(label_idx_0)):int(
            (split_ratio[0] + split_ratio[1]) * len(label_idx_0))],
                            label_idx_1[int(split_ratio[0] * len(label_idx_1)):int(
                                (split_ratio[0] + split_ratio[1]) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((split_ratio[0] + split_ratio[1]) * len(label_idx_0)):],
                             label_idx_1[int((split_ratio[0] + split_ratio[1]) * len(label_idx_1)):])
    else:
        idx_test = np.append(label_idx_0[min(int(split_ratio[0] * len(label_idx_0)), label_number // 2):],
                             label_idx_1[min(int(split_ratio[0] * len(label_idx_1)), label_number // 2):])
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="./datasets/german/",
                label_number=1000, split_ratio=None, val_idx=True):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    #    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
    #        val = idx_features_labels['PurposeOfLoan'].unique()[i]
    #        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

    #    # Normalize LoanAmount
    #    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
    #
    #    # Normalize Age
    #    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
    #
    #    # Normalize LoanDuration
    #    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
    #
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = np.append(label_idx_0[:min(int(split_ratio[0] * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(split_ratio[0] * len(label_idx_1)), label_number // 2)])

    if val_idx:
        idx_val = np.append(label_idx_0[int(split_ratio[0] * len(label_idx_0)):int(
            (split_ratio[0] + split_ratio[1]) * len(label_idx_0))],
                            label_idx_1[int(split_ratio[0] * len(label_idx_1)):int(
                                (split_ratio[0] + split_ratio[1]) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((split_ratio[0] + split_ratio[1]) * len(label_idx_0)):],
                             label_idx_1[int((split_ratio[0] + split_ratio[1]) * len(label_idx_1)):])
    else:
        idx_test = np.append(label_idx_0[min(int(split_ratio[0] * len(label_idx_0)), label_number // 2):],
                             label_idx_1[min(int(split_ratio[0] * len(label_idx_1)), label_number // 2):])
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens
