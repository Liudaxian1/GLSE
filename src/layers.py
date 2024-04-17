import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logmap0
import numpy as np
name2activation = {
    'exp': torch.exp,
    'sig': torch.sigmoid,
    'soft': F.softplus,
    'tanh': torch.tanh,
    'relu': torch.relu,
    'rrelu': torch.rrelu,
    '': lambda x: x
}

class RGCNLayer(nn.Module):
    def __init__(self, rank, n_relations, rel=None, dropout=0.0, self_loop=False, init_size=0.001, bias=False):
        # in_feat: dim
        super(RGCNLayer, self).__init__()
        # self.tf = tf
        self.rank = rank # dim: 40
        self.n_relations = n_relations # n_relation*2 (including inverse relations)
        self.init_size = init_size # 0.001
        self.dropout = dropout # 0.0
        self.self_loop = self_loop # True
        self.bias = nn.Parameter(torch.zeros((1, self.rank))) if bias else None # True (1, dim)
        self.rel, _ = torch.chunk(rel.weight.data, 2, dim=1) # self.rel: (n_relations, dim)

    def forward(self, g):
        """
        :param g:
        :return:
        """
        node_repr = g.ndata['h'].clone() # (n_entities, dim)
        masked_index = torch.masked_select(
            torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(), # .cuda()
            (g.in_degrees(range(g.number_of_nodes())) > 0))
        self.propagate(g)

        if self.self_loop: # True
            node_repr[masked_index, :] = g.ndata['h'][masked_index, :]
        else:
            node_repr = g.ndata['h']
        if self.bias is not None: # True
            node_repr = node_repr + self.bias

        node_repr = F.dropout(node_repr, self.dropout, training=self.training)

        g.ndata['h'] = node_repr
        return g

    def propagate(self, g):
        """
        :param g:
        :return: ['h'] Euclidean
        """
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges):
        r_emb = self.rel.cuda().index_select(0, edges.data['type']).view(-1, self.rank) # (snapSize, dim)
        e_o = edges.src['h'].view(-1, self.rank) # (snapSize, dim)
        msg = e_o + r_emb
        return {'msg': msg} # compute message from edges

    def apply_func(self, nodes):
        h = nodes.data['h']
        norm = nodes.data['norm']
        h = h * norm
        return {'h': h} # perform final norm for the feature 'h'


class HGNNLayer(nn.Module):
    def __init__(self, rank, n_relations, dropout=0.0, reason=None, self_loop=False, init_size=0.001, bias=False):
        # in_feat: dim
        super(HGNNLayer, self).__init__()
        # self.tf = tf
        self.rank = rank # dim: 40
        self.n_relations = n_relations # n_relation*2 (including inverse relations)
        self.init_size = init_size # 0.001
        self.reason_fun = reason # AttH.get_queries()
        self.dropout = dropout # 0.0
        self.self_loop = self_loop # True
        self.bias = nn.Parameter(torch.zeros((1, self.rank))) if bias else None # True (1, dim)

    def forward(self, g):
        """
        :param g:
        :return:
        """
        node_repr = g.ndata['h'].clone() # (n_entities, dim)
        masked_index = torch.masked_select(
            torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(), # .cuda()
            (g.in_degrees(range(g.number_of_nodes())) > 0))
        self.propagate(g)

        if self.self_loop: # True
            node_repr[masked_index, :] = g.ndata['h'][masked_index, :]
        else:
            node_repr = g.ndata['h']
        if self.bias is not None: # True
            node_repr = node_repr + self.bias

        node_repr = F.dropout(node_repr, self.dropout, training=self.training)

        g.ndata['h'] = node_repr
        return g

    def propagate(self, g):
        '''
        :param g:
        :return:
        '''
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges):
        r = edges.data['type'].view(-1, 1) # (snapSize, 1) (including inverse relations)当前时间戳所有边的编号
        e_o = edges.src['h'].view(-1, self.rank) # (snapSize, dim)
        (q, c), _ = self.reason_fun(r, e_o, gc_mode=True) # q: (snapSize, dim) c: (snapSize, 1)
        msg = logmap0(q, c) # (snapSize, dim) Hyperbolic -> Euclidean
        return {'msg': msg} # compute message from edges

    def apply_func(self, nodes):
        h = nodes.data['h']
        norm = nodes.data['norm']
        h = h * norm
        return {'h': h} # perform final norm for the feature 'h'


class AttentionLayer(nn.Module):
    def __init__(self, rank, n_ent, use_time, history_len, init_emb, n_head, dropout=0., layer_norm=False, double_precision=False, init_size=0):
        super(AttentionLayer, self).__init__()
        self.use_time = use_time # ''
        self.dropout = dropout # 0.0
        self.init_emb = init_emb # self.init_ent_emb (n_entities, dim)
        self.history_len = history_len # args.history_len
        self.n_ent = n_ent # n_entities
        self.rank = rank # dim
        self.init_size = init_size # 0.001
        rank_t = rank # dim
        self.multi_head_target = MultiHeadAttention(n_head, rank_t, rank_t//n_head, rank_t//n_head, dropout, layer_norm)
        self.merge = TransformerFFN(rank_t, rank, rank, rank, layer_norm, dropout=dropout) # layer_norm = False

    def forward(self, history_emb):
        if not self.use_time: # True HERE
            q = torch.unsqueeze(self.init_emb.weight, dim=1) # .weight可以计算梯度 (n_entities, 1, dim)
            k = torch.cat(history_emb, dim=1).view(-1, len(history_emb), self.rank) # (n_entities, his_len, dim)
        else:
            k, q = self.time_encoder(history_emb)
        output, attn = self.multi_head_target(q=q, k=k, v=k) # Q, K, V 1层的multi-head_attention
        output = output.squeeze() # (n_entities, 1, dim) -> (n_entities, dim)
        output = self.merge(output, torch.squeeze(q, dim=1)) # (n_entities, dim) -> (n_entities, dim)
        return None, output # (n_entities, dim)


class TransformerFFN(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, layer_norm=False, act='relu', dropout=0.):
        super().__init__()
        print("transformer ffn")
        self.layer_norm = torch.nn.LayerNorm(dim1) if layer_norm else None
        self.fc1 = torch.nn.Linear(dim1, dim3) # dim
        self.fc2 = torch.nn.Linear(dim3, dim4)
        # self.act = nn.ReLU()
        self.act = name2activation[act] # torch.relu
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = x1 + x2  # residual (batch_size, dim)
        res = self.dropout(self.act(self.fc1(x)))
        if self.layer_norm is not None:
            res = self.layer_norm(res)
        res = self.dropout(self.fc2(res))
        return res


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., layer_norm=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        if mask:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual  # residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output, attn