# -*- coding: utf-8 -*-
import math
import torch
from torch.utils.data import DataLoader
from layer import *
from loss import *
from utils import *


class Model(object):
    def __init__(self, data, 
                 encoder, gnn_hidden_channels, gnn_num_layers, 
                 predictor, mlp_hidden_channels, mlp_num_layers, 
                 loss_fn='ce-loss', optimizer='Adam', lr=0.001, dropout=0.0, train_node_emb=False):
        
        device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        self.loss_fn = loss_fn
        self.num_nodes = data.num_nodes
        self.num_node_feats = data.num_features if data.num_features else 0
        self.train_node_emb = train_node_emb
        self.emb = None
        
        if self.num_node_feats:
            self.input_dim = self.num_node_feats
            if self.train_node_emb:
                self.emb = torch.nn.Embedding(self.num_nodes, gnn_hidden_channels)
                self.input_dim += gnn_hidden_channels
        else:
            self.emb = torch.nn.Embedding(self.num_nodes, gnn_hidden_channels)
            self.input_dim = gnn_hidden_channels

        self.encoder = encoder(self.input_dim, gnn_hidden_channels, gnn_hidden_channels, gnn_num_layers, dropout).to(device)
        self.predictor = create_predictor_layer(mlp_hidden_channels, mlp_num_layers, dropout, predictor).to(device)
    
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.emb = self.emb.to(device)
            self.para_list += list(self.emb.parameters())

        if optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)

    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.num_node_feats:
            input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_fn == 'auc_loss':
            loss = loss = auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_fn == 'info_nce_loss':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_fn == 'log_rank_loss':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_fn == 'hinge_auc_loss':
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_fn == 'adaptive_auc_loss' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_fn == 'weighted_auc_loss' and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_fn == 'adaptive_hinge_auc_loss' and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_fn == 'weighted_hinge_auc_loss' and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_fn == 'ce_loss':
            loss = ce_loss(pos_out, neg_out)
        return loss

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        self.encoder.train()
        self.predictor.train()

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=neg_sampler_name,
                                                           num_neg=num_neg)

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)
        edge_weight_margin = split_edge['train']['weight'].to(self.device) if 'weight' in split_edge['train'] else None
        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t)
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data, split_edge, batch_size, evaluator, eval_metric):
        self.encoder.eval()
        self.predictor.eval()
        input_feat = self.create_input_feat(data)

        h = self.encoder(input_feat, data.adj_t)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid', split_edge)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge)
        pos_valid_edge, neg_valid_edge = pos_valid_edge.to(self.device), neg_valid_edge.to(self.device)
        pos_test_edge, neg_test_edge = pos_test_edge.to(self.device), neg_test_edge.to(self.device)

        pos_valid_pred = self.batch_predict(h, pos_valid_edge, batch_size)
        neg_valid_pred = self.batch_predict(h, neg_valid_edge, batch_size)

        h = self.encoder(input_feat, data.adj_t)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_test_pred = self.batch_predict(h, pos_test_edge, batch_size)
        neg_test_pred = self.batch_predict(h, neg_test_edge, batch_size)

        if eval_metric == 'hits':
            results = evaluate_hits(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)
        else:
            results = evaluate_mrr(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)

        return results


def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'DOT':
        return DotPredictor()
    elif predictor_name == 'BIL':
        return BilinearPredictor(hidden_channels)
    elif predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPDOT':
        return MLPDotPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPBIL':
        return MLPBilPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)