import logging
import datetime
import torch.optim
import math
from utils import *
import os.path
from model import GLSE
from optimizer import *
import sys
from config import parser
import torch.nn as nn

def train(args):
    torch.manual_seed(2024)
    save_dir = get_savedir(args.dataset, args.model, args.encoder, args.decoder, args.metrics)
    model_name = "model_d{}-ly{}-his{}-dp{}".format(args.rank, args.n_layers, args.history_len, args.dropout)
    model_path = os.path.join(save_dir, '{}'.format(model_name))
    if args.double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    dataset = load_dataset(args.dataset, load_time=True)
    args.sizes = dataset.get_shape() # (n_entities, n_relations) (including the inverse relations)
    print("\t " + str(dataset.get_shape()))
    train_list = split_by_time(dataset.data["train"]) # [+逆np.array[[s,r,o],...],...]
    valid_list = split_by_time(dataset.data["valid"]) # [+逆np.array[[s,r,o],...],...]
    test_list = split_by_time(dataset.data["test"]) # [+逆np.array[[s,r,o],...],...]

    filters = {}
    filtered_ans_valid = load_all_answers_for_time_filter(dataset.data["valid"])
    filtered_ans_test = load_all_answers_for_time_filter(dataset.data["test"])

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!!!")
    args.device = torch.device("cuda:0" if use_cuda else "cpu")

    # GLSE model
    model = GLSE(args)
    model = nn.DataParallel(model)
    total = count_params(model)
    print("Total number of parameters {}".format(total))
    model.to(args.device)
    model.cuda(args.device)

    # Optimizer
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, optim_method, args.ft_epochs, args.norm_weight, args.valid_freq, args.history_len, args.multi_step, args.topk,
                            args.batch_size, args.neg_sample_size, bool(args.double_neg), args.metrics, use_cuda, args.dropout)

    if args.test:
        # start test#########################
        print("\t ---------------------------Start Testing!---------------------------")
        # True Here
        model.load_state_dict(torch.load(model_path))  # load best model
        model.eval()
        # Test metrics
        print("Evaluation Test Set:")
        _, rank, filter_rank = optimizer.evaluate(train_list + valid_list, test_list, filtered_ans_test, filters, valid_mode=False, epoch=-1)
        test_metrics_raw = compute_metrics(rank)
        test_metrics_filter = compute_metrics(filter_rank)
        print(format_metrics(test_metrics_raw, split="Raw test_" + args.metrics))
        print(format_metrics(test_metrics_filter, split="Time-aware filtering test_" + args.metrics))
    else:
        # start train######################################
        counter = 0
        best_mrr = None
        best_epoch = None
        print("\t ---------------------------Start Training!-------------------------------")
        for epoch in range(args.max_epochs):  # 对于所有的训练轮次
            model.train()
            if use_cuda:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache() # 在每一轮次训练前清空显存参数

            loss, snaps_score = optimizer.epoch(train_list, epoch=epoch) # train_list: [+逆np.array[[s,r,o],...],...]
            print("\t Epoch {} | average train loss: {:.4f}".format(epoch, loss))

            if math.isnan(loss.item()):
                break
            # validation#####################################
            model.eval()
            valid_loss, ranks, filter_ranks = optimizer.evaluate(train_list, valid_list, filtered_ans_valid, filters, epoch=epoch, valid_mode=True)
            print("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))

            if (epoch + 1) % args.valid_freq == 0:
                valid_metrics = compute_metrics(ranks)
                print(format_metrics(valid_metrics, split="valid_" + args.metrics))
                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = epoch
                    print("\t Saving model at epoch {} in {}".format(epoch, save_dir))
                    torch.save(model.cpu().state_dict(), model_path)
                    if use_cuda:
                        model.cuda()
                else:
                    counter += 1
                    if counter == args.patience:  # cannot early stop
                        print("\t Early stopping")
                        break
                    elif counter == args.patience // 2:
                        pass
        print("\t ---------------------------Optimization Finished!---------------------------")
        if best_mrr:  # os.path.exists(model_name)
            print("\t Saving best model at epoch {}".format(best_epoch))
    return None

if __name__ == "__main__":
    train(parser)
    sys.exit()
