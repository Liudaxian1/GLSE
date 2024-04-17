import numpy as np
import torch
from tqdm import tqdm
import random
from utils import build_sub_graph
from utils import compute_metrics, get_ranking, construct_snap
from torch import nn


class KGOptimizer(object):
    def __init__(self, model, optimizer, ft_epochs, norm_weight, valid_freq, history_len, multi_step, topk, batch_size, neg_sample_size,
                 double_neg=False, metrics='raw', use_cuda=False, dropout=0., verbose=True, grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.grad_norm = grad_norm
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean') # false->0, 最后求均值
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.module.sizes[0]
        self.n_relations = model.module.sizes[1]
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.valid_freq = valid_freq
        self.multi_step = multi_step
        self.topk = topk
        self.history_len = history_len
        self.metrics = metrics
        self.dropout = dropout
        self.ft_epochs = ft_epochs
        self.norm_weight = norm_weight

    def calculate_loss(self, out_g, ent_embs, epoch=-1):
        loss = torch.zeros(1).cuda().to(self.device) if self.use_cuda else torch.zeros(1)
        mean_score = None
        scores, factors = self.model.module.reason(out_g, ent_embs, eval_mode=True, epoch=epoch) # list(所有可能实体),行
        truth = out_g[:, 2] # (batch_size, 1)
        for idx in range(len(ent_embs)):
            loss += self.loss_fn(scores[idx], truth) # scores: [(batch_size, n_entities), ...] truth: (batch_size, 1)
        return loss, mean_score

    def epoch(self, train_list, epoch=-1):
        losses = []
        idx = [_ for _ in range(len(train_list))] # 时间戳编号 train_list: [+逆np.array[[s,r,o],...],...]
        random.shuffle(idx) # 打乱训练过程的时间戳顺序
        score_of_snap = np.zeros(len(train_list))
        for train_sample_num in tqdm(idx): # 对于训练集的每一个时间戳
            if train_sample_num == 0:
                continue
            output = train_list[train_sample_num] # train的当前时间戳所有事实: +逆np.array[[s,r,o],...]
            if train_sample_num - self.history_len < 0: # input_list: 历史图谱序列
                input_list = train_list[0: train_sample_num]
            else:
                input_list = train_list[train_sample_num - self.history_len:train_sample_num]

            history_g_list = [build_sub_graph(self.n_entities, self.n_relations, snap, self.use_cuda, self.device, self.dropout)
                              for snap in input_list] # [dgl list] n_entities: 全局的实体数目 n_relations: 全局的关系数目*2
            output = torch.from_numpy(output).long().cuda().to(self.device) if self.use_cuda else torch.from_numpy(output).long()
            # loss&GD
            evolve_ent_embs = self.model.module.evolve(history_g_list) # history_g_list: [historical dgl list] evolve_ent_emb: ( , (n_entities, dim))
            loss, mean_score = self.calculate_loss(output, evolve_ent_embs, epoch=epoch) # output: (batch_size, 3) evolve_ent_emb[-1]: (n_entities, dim)

            # optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())
        return np.mean(losses), score_of_snap

    def evaluate(self, history_list, test_list, filtered_ans_list, filters, valid_mode=False, epoch=-1, multi_step_test=False, topk_test=0):
        valid_losses = []
        valid_loss = None
        ranks = []
        filter_ranks = []
        input_list = [snap for snap in history_list[-self.history_len:]] # input_list: [+逆np.array[[s,r,o],...],...]
        with torch.no_grad():
            for time_idx, test_snap in enumerate(tqdm(test_list)): # 对于测试集的每一个时间戳
                history_g_list = [build_sub_graph(self.n_entities, self.n_relations, g, self.use_cuda, self.device) for g in input_list] # [historical dgl list]
                test_triples = torch.LongTensor(test_snap).cuda().to(self.device) if self.use_cuda else torch.LongTensor(test_snap) # (batch_size, 3)
                evolve_ent_embs = self.model.module.evolve(history_g_list) # evolve_ent_emb[-1]: (n_entities, dim)
                if valid_mode: # True
                    loss, mean_score = self.calculate_loss(test_triples, evolve_ent_embs, epoch=epoch) # test_triples: (batch_size, 3) evolve_ent_emb[-1]: (n_entities, dim)
                    valid_losses.append(loss.item())
                if (epoch + 1) % self.valid_freq == 0 or not valid_mode:
                    scores, _ = self.model.module.reason(test_triples, evolve_ent_embs, eval_mode=True, epoch=epoch) # scores: (batch_size, n_entities)
                    scores = [_.unsqueeze(2) for _ in scores] # (batch_size, n_entities, 1)
                    scores = torch.cat(scores, dim=2) # (batch_size, n_entities, his_len)
                    scores = torch.softmax(scores, dim=1) # (batch_size, n_entities, his_len)
                    score = torch.sum(scores, dim=-1) # (batch_size, n_entities)
                    _, _, rank, filter_rank = get_ranking(test_triples, score, filtered_ans_list[time_idx], filters, self.metrics, batch_size=self.batch_size)
                    ranks.append(rank)
                    filter_ranks.append(filter_rank)
                    input_list.pop(0)
                    input_list.append(test_snap)
            if valid_losses:
                valid_loss = np.mean(valid_losses)
            if ranks:
                ranks = torch.cat(ranks)
            if filter_ranks:
                filter_ranks = torch.cat(filter_ranks)
            return valid_loss, ranks, filter_ranks
