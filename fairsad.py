import torch.nn as nn
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import *
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F


class FairSAD(torch.nn.Module):
    def __init__(self, train_args):
        super(FairSAD, self).__init__()
        self.args = train_args
        # model of FairSAD: encoder, masker, classifier
        self.encoder = DisGCN(nfeat=self.args.nfeat,
                                nhid=self.args.hidden,
                                nclass=self.args.nclass,
                                chan_num=4,
                                layer_num=2,
                                dropout=self.args.dropout
                                ).to(self.args.device)
        self.masker = channel_masker(train_args.hidden).to(self.args.device)
        self.classifier = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)

        self.per_channel_dim = train_args.hidden // train_args.channels
        self.channel_cls = nn.Linear(self.per_channel_dim, train_args.channels).to(self.args.device)

        # optumizer
        self.optimizer_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()) + list(self.masker.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        self.optimizer_c = torch.optim.Adam(list(self.channel_cls.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        # loss function
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        self.criterion_mask = FeatCov()

        self.encoder.init_parameters()
        self.encoder.init_edge_weight()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        h = self.masker(h)
        output = self.classifier(h)
        return h, output

    def train_fit(self, data, epochs, **kwargs):
        # parsing parameters
        alpha = kwargs.get('alpha', None)
        beta = kwargs.get('beta', None)
        pbar = kwargs.get('pbar', None)

        best_res_val = 0.0
        save_model = 0

        # training encoder, assigner, classifier
        for epoch in range(epochs):
            self.encoder.train()
            self.masker.train()
            self.classifier.train()
            self.channel_cls.train()

            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()

            h = self.encoder(data.features, data.edge_index)
            h = self.masker(h)
            output = self.classifier(h)
            # output = self.encoder.predict(h)

            # downstream tasks loss
            loss_cls_train = self.criterion_bce(output[data.idx_train],
                                                data.labels[data.idx_train].unsqueeze(1).float())

            # channel identification loss
            loss_chan_train = 0
            for i in range(self.args.channels):
                chan_output = self.channel_cls(h[:, i*self.per_channel_dim:(i+1)*self.per_channel_dim])
                chan_tar = torch.ones(chan_output.shape[0], dtype=int)*i
                chan_tar = chan_tar.to(self.args.device)
                loss_chan_train += self.criterion_mul_cls(chan_output, chan_tar)

            # distance correlation loss
            loss_disen_train = 0
            len_per_channel = int(h.shape[1] / self.args.channels)
            for i in range(self.args.channels):
                for j in range(i + 1, self.args.channels):
                    loss_disen_train += self.criterion_dc(
                        h[data.idx_train, i * len_per_channel:(i + 1) * len_per_channel],
                        h[data.idx_train, j * len_per_channel:(j + 1) * len_per_channel])

            # masker loss
            loss_mask_train = self.criterion_mask(h[data.idx_train], data.sens[data.idx_train])

            loss_train = loss_cls_train + alpha * (loss_chan_train + loss_disen_train) + beta * loss_mask_train

            loss_train.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()

            # evaluating encoder, assigner, classifier
            self.encoder.eval()
            self.classifier.eval()

            if epoch % 10 == 0:
                self.masker.eval()
                h = self.encoder(data.features, data.edge_index)
                h = self.masker(h)
                y_output_val = self.classifier(h)
                y_output_val = y_output_val.detach()
                y_pred_val = (y_output_val.squeeze() > 0).type_as(data.sens)
                acc_val = accuracy_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                roc_val = roc_auc_score(data.labels[data.idx_val].cpu(), y_output_val[data.idx_val].cpu())
                f1_val = f1_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                parity, equality = fair_metric(y_pred_val[data.idx_val].cpu().numpy(),
                                               data.labels[data.idx_val].cpu().numpy(),
                                               data.sens[data.idx_val].cpu().numpy())

                res_val = acc_val + roc_val - parity - equality

                if res_val > best_res_val:
                    best_res_val = res_val
                    """
                        evaluation
                    """
                    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    roc_test = roc_auc_score(data.labels[data.idx_test].cpu(), y_output_val[data.idx_test].cpu())
                    f1_test = f1_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    parity_test, equality_test = fair_metric(y_pred_val[data.idx_test].cpu().numpy(),
                                                   data.labels[data.idx_test].cpu().numpy(),
                                                   data.sens[data.idx_test].cpu().numpy())
                    save_model = epoch

            if pbar is not None:
                pbar.set_postfix({'train_total_loss': "{:.2f}".format(loss_train.item()),
                                  'cls_loss':  "{:.2f}".format(loss_cls_train.item()),
                                  'disen_loss': "{:.2f}".format(loss_disen_train.item()),
                                  'mask_loss': "{:.2f}".format(loss_mask_train.item()),
                                  'val_loss': "{:.2f}".format(res_val), 'save model': save_model})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return roc_test, f1_test, acc_test, parity_test, equality_test

class DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = self.out_dim // self.channels
        self.reduce = reduce

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(Linear(in_channels=self.per_channel_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
            else:
                self.conv_layers.append(Linear(in_channels=self.in_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))

    def get_reddim_k(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feat = self.lin_layers[k](x)
            z_feats.append(z_feat)
        return z_feats

    def get_k_feature(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feats.append(x)
        return z_feats

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[1], "axis dimension in direction 1 need to be equal to channels number"
        if self.reduce:
            z_feats = self.get_reddim_k(x)
        else:
            z_feats = self.get_k_feature(x)
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight[:, k])
            out = self.propagate(edge_index_copy, x=c_temp)
            if self.bias_list is not None:
                out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
        output = torch.cat(c_feats, dim=1)
        return output

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, chan_num, layer_num, dropout=0.5):
        super(DisGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout_rate = dropout
        self.chan_num = chan_num
        self.layer_num = layer_num
        self.edge_weight = None

        self.assigner = NeiborAssigner(nfeat, chan_num)
        self.disenlayers = nn.ModuleList()
        for i in range(layer_num-1):
            if i == 0:
                self.disenlayers.append(DisenLayer(nfeat, nhid, chan_num))
            else:
                self.disenlayers.append(DisenLayer(nhid, nhid, chan_num))
        self.dropout = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)
    
    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        edge_weight = self.assigner(feats_pair.detach())
        for layer in self.disenlayers:
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        return x

class NeiborAssigner(nn.Module):
    def __init__(self, nfeats, channels):
        super(NeiborAssigner, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        alpha_score = self.layers(features_pair)
        alpha_score = torch.softmax(alpha_score, dim=1)
        return alpha_score

class channel_masker(nn.Module):
    def __init__(self, hid_num):
        super(channel_masker, self).__init__()
        self.hid_num = hid_num
        self.weights = nn.Parameter(torch.distributions.Uniform(0, 1).sample((hid_num, 2)))

    def forward(self, x):
        mask = F.gumbel_softmax(self.weights, tau=1, hard=False)[:, 0]
        x = x * mask
        return x