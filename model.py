# -*- coding: utf-8 -*-
import math
import torch
import time
from torch.utils.data import DataLoader
from torch_cluster import random_walk
from layer import *
from loss import *
from utils import *


class Model(object):
    def __init__(self, data, 
                 encoder, gnn_hidden_channels, gnn_num_layers, 
                 predictor, mlp_hidden_channels, mlp_num_layers, 
                 loss_fn='ce-loss', optimizer='Adam', lr=0.001, dropout=0.0, train_node_emb=False):
        
        device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

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

        self.encoder = encoder(self.input_dim, gnn_hidden_channels, gnn_hidden_channels, gnn_num_layers, dropout).to(self.device)
        self.predictor = create_predictor_layer(mlp_hidden_channels, mlp_num_layers, dropout, predictor).to(self.device)
    
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.emb = self.emb.to(self.device)
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



def run_link_prediction(data, split_edge, model, evaluator, logger, eval_metric, batch_size=65536,
                        runs=10, epochs=1000, lr=0.001, neg_sampler='random', num_neg=1, eval_steps=1,
                        log_steps=1, random_walk_augment=False, walk_start_type='edge'):

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data = data.to(device)

    if eval_metric == 'hits':
        loggers = {
            'Hits@20': logger(runs),
            'Hits@50': logger(runs),
            'Hits@100': logger(runs),
        }

    elif eval_metric == 'mrr':
        loggers = {
            'MRR': logger(runs),
        }

    if random_walk_augment:
        rw_row, rw_col, _ = data.adj_t.coo()
        if walk_start_type == 'edge':
            rw_start = torch.reshape(split_edge['train']['edge'], (-1,)).to(device)
        else:
            rw_start = torch.arange(0, data.num_nodes, dtype=torch.long).to(device)

    cur_lr = lr
    for run in range(runs):
        print(f'run: {run}')
        model.param_init()
        start_time = time.time()

        for epoch in range(1, 1 + epochs):
            print(f'epoch: {epoch}')
            if random_walk_augment:
                walk = random_walk(rw_row, rw_col, rw_start, walk_length=5)
                pairs = []
                weights = []
                for j in range(5):
                    pairs.append(walk[:, [0, j + 1]])
                    weights.append(torch.ones((walk.size(0),), dtype=torch.float) / (j + 1))
                pairs = torch.cat(pairs, dim=0)
                weights = torch.cat(weights, dim=0)
                # remove self-loop edges
                mask = ((pairs[:, 0] - pairs[:, 1]) != 0)
                split_edge['train']['edge'] = torch.masked_select(pairs, mask.view(-1, 1)).view(-1, 2)
                split_edge['train']['weight'] = torch.masked_select(weights, mask)

            print(f'Training Started')
            loss = model.train(data, split_edge,
                               batch_size=batch_size,
                               neg_sampler_name=neg_sampler,
                               num_neg=num_neg)
            print(f'Training completed')
            print(f'Loss: {loss}\n\n')
            if epoch % eval_steps == 0:
                results = model.test(data, split_edge,
                                     batch_size=batch_size,
                                     evaluator=evaluator,
                                     eval_metric=eval_metric)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Learning Rate: {cur_lr:.4f}, '
                                    f'Valid: {100 * valid_res:.2f}%, '
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                    print('---')
                    print(
                        f'Training Time Per Epoch: {spent_time / eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, last_best=False)
            print('-'*100)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(last_best=False)
