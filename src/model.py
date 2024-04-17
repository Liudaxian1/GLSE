from abc import ABC, abstractmethod
from layers import *
from utils import givens_rotations, givens_reflection, mobius_add, expmap0, project, hyp_distance_multi_c, logmap0, operations

class KGModel(nn.Module, ABC):
    def __init__(self, sizes, rank, dropout, gamma, bias, init_size, use_cuda=False):
        super(KGModel, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.rel = nn.Embedding(sizes[1], rank) # (n_relations*2, dim)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1))
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1))

    def forward(self, queries, ent_emb, eval_mode=False, rel_emb=None, c=None):
        lhs_e, lhs_biases = self.get_queries(queries, ent_emb) # lhs_e: ((batch_size, dim), (batch_size, 1))
        rhs_e, rhs_biases = self.get_rhs(queries, ent_emb, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode) # (batch_size, n_entities) scores based on hyperbolic distance

        factors = self.get_factors(queries, ent_emb)
        return predictions, factors # predictions: (batch_size, n_entities)

    @abstractmethod
    def get_queries(self, queries, ent_emb):
        pass

    @abstractmethod
    def get_rhs(self, queries, ent_emb, eval_mode):
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn': # True Here
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries, ent_emb):
        head_e = ent_emb[queries[:, 0]]
        rel_e = self.rel(queries[:, 1])
        rhs_e = ent_emb[queries[:, 2]]
        return head_e, rel_e, rhs_e


class BaseH(KGModel):
    """Trainable curvature for each relationship."""
    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.bias, args.init_size)

        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank)) # (n_relations, dim * 2)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank)) - 1.0
        self.multi_c = args.multi_c

        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1))
        else:
            c_init = torch.ones((1, 1))
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, ent_emb=None, eval_mode=False):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return ent_emb, self.bt.weight
        else:
            return ent_emb[queries[:, 2]], self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    def get_c(self):
        """
        for single c
        :return: parameter of curvature
        """
        if self.multi_c:
            return self.c
        else: # True Here
            return self.c.repeat(self.sizes[1], 1) # (n_relations*2, 1)


class AttH(BaseH):
    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank) # (n_relations*2, dim*2)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank)) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank) # (n_relations*2, dim) att vec
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank))
        self.act_att = nn.Softmax(dim=1)
        self.scale = nn.Parameter(torch.Tensor([1. / np.sqrt(self.rank)]), requires_grad=False)

    def get_queries(self, queries, ent_emb, gc_mode=False):
        if gc_mode:
            queries = queries.view(-1, 1) # (snapSize, 1) 当前时间戳所有边的类型编号
            s = torch.zeros_like(queries) # make no sense
            queries = torch.cat((s, queries), 1) # (snapSize, 2) 有用的是queries[:, 1]
            head = ent_emb # (snapSize, dim)
        else:
            head = ent_emb[queries[:, 0]]
        c_p = self.get_c() # (n_relations*2, 1)
        c = F.softplus(c_p[queries[:, 1]]) # (snapSize, 1)
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1) # rot_mat: (snapSize, dim) ref_mat: (snapSize, dim)

        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank)) # (snapSize, 1, dim)
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank)) # (snapSize, 1, dim)
        cands = torch.cat([ref_q, rot_q], dim=1) # (snapSize, 2, dim)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank)) # (snapSize, 1, dim)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True) # (snapSize, 2, dim)
        att_weights = self.act_att(att_weights) # (snapSize, 2, dim)
        att_q = torch.sum(att_weights * cands, dim=1) # (snapSize, 2, dim) -> (snapSize, dim)

        lhs = expmap0(att_q, c) # (snapSize, dim) Euclidean -> Hyperbolic
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # (snapSize, dim)
        rel = expmap0(rel, c) # (snapSize, dim) Euclidean -> Hyperbolic
        res = project(mobius_add(lhs, rel, c), c) # (snapSize, dim)
        return (res, c), self.bh(queries[:, 0]) # res: (snapSize, dim) c: (snapSize, 1)


class GLSE(AttH):
    def __init__(self, args):
        super(GLSE, self).__init__(args)
        self.device = args.device
        self.n_layers = args.n_layers
        self.history_len = args.history_len
        self.en_dropout = args.dropout
        self.de_dropout = args.de_dropout
        self.up_dropout = args.up_dropout
        self.model_name = args.model

        # ent emb
        self.init_ent_emb = nn.Embedding(self.sizes[0], self.rank) # 所有实体的动态嵌入 (n_entities, dim)
        self.init_ent_emb.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank)) # 动态嵌入初始化

        # dynamic-----------------------
        self.entity_cell = nn.GRUCell(self.rank, self.rank)
        self.h = None
        assert args.n_layers > 0
        self.update = AttentionLayer(self.rank, args.sizes[0], args.use_time, self.history_len, self.init_ent_emb, args.n_head,
                                     args.up_dropout, args.layer_norm, args.double_precision, self.init_size)
        # dynamic cell
        self.layers_h = nn.ModuleList()
        self.layers_e = nn.ModuleList()
        self.build_layers(args) # GCLayers
        if self.model_name == 'two': # two
            self.s_hp = nn.Parameter(torch.Tensor([args.s_hp]), requires_grad=False) # -1
            self.s_delta_ind = args.s_delta_ind # True
            if args.s_hp < 0:
                if args.s_delta_ind: # Here
                    self.delta_l = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True) # (n_entities, 1)
                    self.delta_r = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True) # (n_entities, 1)
                else:
                    self.delta = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
            self.score_comb = operations[args.s_comb] # torch.maximum default: str"max" Method to combine two scores
            self.score_softmax = args.s_softmax # False
            self.s_dropout = args.s_dropout # 0.0
            self.reason_dropout = args.reason_dropout # 0.0

    def build_layers(self, args):
        for i in range(self.n_layers): # n_layers: 2
            self.layers_h.append(
                HGNNLayer(self.rank, self.sizes[1], self.en_dropout,
                        self.get_queries, args.en_loop, self.init_size, args.en_bias))
            self.layers_e.append(
                RGCNLayer(self.rank, self.sizes[1], self.rel, self.en_dropout,
                          args.en_loop, self.init_size, args.en_bias))

    def evolve(self, g_list): # g_list: [historical dgl list]
        self.h = self.init_ent_emb.weight.clone() # (n_entities, dim)
        evolve_embs = []
        for idx in range(len(g_list)):
            history_emb = []
            for i, g in enumerate(g_list[idx:]): # 对于历史中的每一个时间戳子图
                g = g.to(self.device) # dgl object
                hidden_h = self.snap_forward_h(g, self.h) # (n_entities, dim) concurrent self.h
                hidden_e = self.snap_forward_e(g, self.h) # (n_entities, dim) concurrent self.h
                hidden = self.update([hidden_h, hidden_e])[-1]
                self.h = self.entity_cell(self.h, hidden)
                history_emb.append(self.h) # (n_entities, his_len, dim)
            evolve_snap = self.update(history_emb)
            evolve_snap = evolve_snap[-1] # (n_entities, dim)
            evolve_embs.append(evolve_snap)
        evolve_embs.reverse()
        return evolve_embs # entity embeddings at each timestamp

    def snap_forward_h(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id] # (n_entities, dim)
        for i, layer in enumerate(self.layers_h):
            layer(g)
        return g.ndata.pop('h')

    def snap_forward_e(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id] # (n_entities, dim)
        for i, layer in enumerate(self.layers_e):
            layer(g)
        return g.ndata.pop('h')

    def comb_score(self, queries, old_score, new_score, act=torch.sigmoid):
        """
        :param queries: (batch_size, 3)
        :param old_score: (batch_size, n_entities)
        :param new_score: (batch_size, n_entities)
        :param act: w(0, 1)
        :return:
        """
        if self.score_softmax: # False
            old_score = torch.softmax(old_score, 1, old_score.dtype)
            new_score = torch.softmax(new_score, 1, new_score.dtype)
        if self.s_hp[0] < 0: # True
            if self.s_delta_ind: # True
                w1 = self.delta_l[queries[:, 0]] # (batch_size, 1)
                w2 = self.delta_r[queries[:, 2]] # (batch_size, 1)
            else:
                w1 = self.delta[queries[:, 0]]
                w2 = self.delta[queries[:, 2]]
            if act:
                w1 = act(w1) # (batch_size, 1)
                w2 = act(w2) # (batch_size, 1)
            w = self.score_comb(w1, w2) # torch.maximum
            w = F.dropout(w, self.up_dropout, training=self.training) # (batch_size, 1) training=self.training是一种固定用法，与model.train()/eval()保持一致
        else:
            w = self.s_hp.repeat(queries.shape[0], 1)
        score = w * new_score + (1 - w) * old_score # (batch_size, n_entities)
        return score # (batch_size, n_entities)

    def reason(self, queries, ent_embs, eval_mode=False, epoch=1000, rel_emb=None, c=None):
        '''
        :param queries: (batch_size, 3) batch_size is the size of a timestamp
        :param ent_embs: [(n_entities, dim), ...]
        :param eval_mode: True
        :return:
        '''
        score_list = []
        new_factors, old_factors = None, None
        for idx in range(len(ent_embs)):
            if self.model_name == 'two' and self.s_hp != 0: # True Here
                new_ent_emb = F.dropout(ent_embs[idx], self.reason_dropout, training=self.training) # (n_entities, dim)
                init_ent_emb = self.init_ent_emb.weight # (n_entities, dim)
                new_score, new_factors = self.forward(queries, new_ent_emb, eval_mode=eval_mode) # (batch_size, n_entities)
                old_score, old_factors = self.forward(queries, init_ent_emb, eval_mode=eval_mode) # (batch_size, n_entities)
                score = self.comb_score(queries, old_score, new_score) # (batch_size, n_entities)
                score_list.append(score) # score: (batch_size, n_entities)
            else:
                score, factor = self.forward(queries, ent_embs[idx], eval_mode=eval_mode)
                score_list.append(score)
        return score_list, (old_factors, new_factors)