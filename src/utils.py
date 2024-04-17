import os
import torch
import dgl
import torch.nn.functional as F
import numpy as np
import pickle as pkl
LOG_DIR = '../results/'
DATA_PATH = '../data/'
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
operations = {
    'add': torch.add,
    'mul': lambda x, y: x * y,
    'unary': lambda x, y: x,
    'div': lambda x, y: x / y.clamp_max(-1e-15) if y < 0 else x / y.clamp_min(1e-15),
    'max': torch.maximum,
    'min': torch.minimum,
    'mean': lambda x, y: (x + y) / 2
}
activations = {
    'exp': torch.exp,
    'sig': torch.sigmoid,
    'soft': F.softplus,
    'tanh': torch.tanh,
    '': None
}


def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2)) # (n x d/2 x 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)    # !sin,cos->单位化二维向量
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)  # !乘变换矩阵
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def _lambda_x(x, c):
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - c * x_sqnorm).clamp_min(MIN_NORM)


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def expmap(u, p, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = ( tanh(sqrt_c / 2 * _lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm))
    gamma_1 = mobius_add(p, second_term, c)
    return gamma_1


def logmap(p1, p2, c):
    sub = mobius_add(-p1, p2, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(p1, c)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)     # murp: (0, 1-1e-5)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)     # murp: (0, 1-1e-5)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_matvec(m, x, c, b=None):     # HNN?
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    if b is not None:
        res = project(mobius_add(res, b, c), c)
    return res


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    if vnorm.min().item() < 1e-15:
        print("error in expmap0")
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


class KGDataset(object):
    def __init__(self, name, dir=None, load_time=True):
        self.name = name
        self.data = {}  # dict
        self.relation_dict = {}
        self.entity_dict = {}
        self.path = dir
        self.path = os.path.join(dir, self.name)
        self.n_entities = -1
        self.n_relations = -1
        self.load(load_time) # 初始化调用

    def load(self, load_time=False):
        entity_path = os.path.join(self.path, 'entity2id.txt')
        relation_path = os.path.join(self.path, 'relation2id.txt')
        entity_dict = read_dictionary(entity_path)
        relation_dict = read_dictionary(relation_path)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        self.n_entities = len(entity_dict)
        self.n_relations = len(relation_dict)*2
        splits = ["train", "test", "valid"]
        for split in splits:
            data_path = os.path.join(self.path, split+'.txt')
            self.data[split] = np.array( # {self.data["train"]: +逆np.array[[s,r,o,t],...],}
                self.read_examples(data_path, entity_dict, relation_dict, load_time=load_time))

    def get_shape(self):
        """
        Returns KG dataset shape.
        :return: int, int; n_rel = len(rel_dict) * 2(逆关系增强)
        """
        return self.n_entities, self.n_relations

    def get_filters(self):
        """
        Return filter dict to compute ranking metrics in the filtered setting.
        :return dict={(e,r):有序list(不重复)}
        """
        filters_file = open(os.path.join(self.path, "to_skip.pickle"), "rb")
        to_skip = pkl.load(filters_file)
        filters_file.close()
        filters = to_skip['rhs']
        filters.update(to_skip['lhs'])
        return filters

    # +逆元组
    def read_examples(self, dataset_file, entity_dict=None, relation_dict=None, load_time=False):
        """
        :param dataset_file:
        :param entity_dict:
        :param relation_dict:
        :param load_time:
        :return: np.array(int64)
        """
        # print("path:{}", dataset_file)
        dataset_file = dataset_file.replace('\\', '/')
        examples = []
        n_rels = self.n_relations//2
        print(dataset_file)
        with open(dataset_file, "r") as lines:
            for line in lines:
                line = line.strip().split('\t')
                s, r, o, st = line[:4]
                if load_time:
                    examples.append([s, r, o, st])
                    examples.append([o, int(r) + n_rels, s, st])
                else:
                    examples.append([s, r, o])
                    examples.append([o, int(r) + n_rels, s])
        # +逆
        return np.array(examples).astype("int64") # +逆np.array[[s,r,o,t],...]


def read_dictionary(filename):
    d = {}
    filename = filename.replace('\\', '/')
    with open(filename, "r", encoding='utf-8') as lines:
        for line in lines:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def load_dataset(dataset, load_time, debug=False):
    if dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return KGDataset(dataset, load_time)
    elif dataset in ['YAGO', 'WIKI', 'ICEWS18', 'ICEWS14', "GDELT", "ICEWS14s", 'ICEWS05-15']:
        return KGDataset(dataset, DATA_PATH, load_time)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def add_subject(e1, e2, r, d, n_relations):
    if not e2 in d:
        d[e2] = {}
    if not r+n_relations in d[e2]:
        d[e2][r+n_relations] = set()
    d[e2][r+n_relations].add(e1)


def add_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def get_all_answer(total_data, n_relations):
    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_ans, n_relations)
        add_object(s, o, r, all_ans)
    return all_ans


def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        exmaple = data[i]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(exmaple[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list # [+逆np.array[[s,r,o],...],...]


def get_all_answers_in_snap(snap_list, n_relations):
    all_ans_list = []
    for snap in snap_list:
        all_ans_t = get_all_answer(snap, n_relations)
        all_ans_list.append(all_ans_t)
    return all_ans_list


def load_all_answers_for_filter(total_data):
    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        add_object(s, o, r, all_ans)
    return all_ans


def load_all_answers_for_time_filter(total_data):
    all_ans_list = []
    all_snap = split_by_time(total_data) # [+逆np.array[[s,r,o],...],...]
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap)
        all_ans_list.append(all_ans_t)
    return all_ans_list


def get_savedir(dataset, model, encoder, decoder, metrics):
    save_dir = os.path.join(LOG_DIR, dataset, 'checkpoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def build_sub_graph(n_entities, n_relations, triples, use_cuda, device, dropout=0.): # triples:Gt
    def comp_deg_norm(g):
        out_deg = g.out_degrees(range(g.number_of_nodes())).float()
        out_deg[torch.nonzero(out_deg == 0).view(-1)] = 1
        norm = 1.0 / out_deg
        return norm

    if dropout > 0: # False
        rand_triples = triples[torch.randperm(triples.shape[0]), :]
        triples = rand_triples[:- int(rand_triples.shape[0] * dropout)]
    src, rel, dst = triples.transpose()

    g = dgl.graph((src, dst), num_nodes=n_entities, device=device)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, n_entities, dtype=torch.long, device=device).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel).cuda() if use_cuda else torch.LongTensor(rel) # (snapSize, 1) (including inverse relations)
    return g


def count_params(model):
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l


def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    target_pos = torch.zeros_like(indices)
    target_pos[indices == target.view(-1, 1)] = 1
    target_index = torch.nonzero(target_pos)
    return target_index[:, 1].view(-1)


def sort_and_rank_atth(scores, target):
    ranks = torch.zeros(len(target), 1).cuda()
    ranks += torch.sum((scores >= target).float(), dim=1)
    return ranks


def get_ranking(test_triples, score, filtered_ans_list, filters, metric_type='time_filter', batch_size=1000):
    num_triples = len(test_triples) # 测试集batch的事实数目
    n_batch = (num_triples + batch_size - 1) // batch_size
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = min(num_triples, (idx + 1) * batch_size)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        filter_score_batch = filter_score(triples_batch, score_batch, filtered_ans_list)
        filter_rank.append(sort_and_rank(filter_score_batch, target))
    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def filter_score(test_triples, score, filtered_ans_list):
    if filtered_ans_list is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(filtered_ans_list[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000
    return score


def compute_metrics(ranks):
    mean_rank = torch.mean(ranks.float()).item()
    mean_reciprocal_rank = torch.mean(1. / ranks.float()).item()
    hits_at = torch.FloatTensor((list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        (1, 3, 10)
    ))))
    return {'MR': mean_rank, 'MRR': mean_reciprocal_rank, 'hits@[1,3,10]': hits_at}


def format_metrics(metrics, split):
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.4f} | ".format(metrics['MRR'])
    result += "H@1: {:.4f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.4f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.4f}".format(metrics['hits@[1,3,10]'][2])
    return result