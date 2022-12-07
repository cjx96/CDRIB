import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.trainer import CrossTrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper
import json
import codecs
import copy
# torch.cuda.set_device(12)

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--dataset', type=str, default='game_video', help='cloth_sport, game video')

# model part
parser.add_argument('--model', type=str, default="CDRIB", help="The model name.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=3, help='GNN layer.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--lambda', type=float, default=0.9)
parser.add_argument('--bce', dest='bce', action='store_true', default=False)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--beta', type=float, default=1.5)
# train part
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--user_batch_size', type=int, default=64, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--test_sample_number', type=int, default=999)

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])

# load data adj-matrix; Now sparse tensor ,But not setting in gpu.
if "CDRIB" in opt["model"]:
    filename  = opt["dataset"]
    source_graph = "../dataset/" + filename + "/train.txt"
    source_G = GraphMaker(opt, source_graph)
    source_UV = source_G.UV
    source_VU = source_G.VU
    source_adj = source_G.adj

    filename = filename.split("_")
    filename = filename[1] + "_" + filename[0]
    target_train_data = "../dataset/" + filename + "/train.txt"
    target_G = GraphMaker(opt, target_train_data)
    target_UV = target_G.UV
    target_VU = target_G.VU
    target_adj = target_G.adj
    print("graph loaded!")

    if opt["cuda"]:
        source_UV = source_UV.cuda()
        source_VU = source_VU.cuda()
        source_adj = source_adj.cuda()

        target_UV = target_UV.cuda()
        target_VU = target_VU.cuda()
        target_adj = target_adj.cuda()


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)


print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
source_valid_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 3)
source_test_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
target_valid_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 4)
target_test_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)

print("source_user_num", opt["source_user_num"])
print("target_user_num", opt["target_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("source train data : {}, target train data {}, source test data : {}, target test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(source_test_batch.test_data),len(target_test_batch.test_data)))
opt["shared_user"] = min(source_test_batch.MIN_USER, target_test_batch.MIN_USER) + 1
opt["source_shared_user"] = source_test_batch.MAX_USER + 1
opt["target_shared_user"] = target_test_batch.MAX_USER + 1
print("shared users id: " + str(opt["shared_user"]))
print("test users {}, {}".format(source_test_batch.MAX_USER - source_test_batch.MIN_USER + 1 , target_test_batch.MAX_USER - target_test_batch.MIN_USER + 1))
# model
if not opt['load']:
    trainer = CrossTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = CrossTrainer(opt)
    trainer.load(model_file)

s_dev_score_history = [0]
t_dev_score_history = [0]

current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

best_s_hit = -1
best_s_ndcg = -1
best_t_hit = -1
best_t_ndcg = -1

# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct_graph(batch, source_UV, source_VU, target_UV, target_VU, source_adj, target_adj, epoch)
        train_loss += loss

    duration = time.time() - start_time
    train_loss = train_loss/len(train_batch)
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss, duration, current_lr))

    if epoch<10 or epoch % 5:
        continue

    # eval model
    print("Evaluating on dev set...")
    trainer.model.eval()

    trainer.evaluate_embedding(source_UV, source_VU, target_UV, target_VU, source_adj, target_adj,epoch)

    def predict(dataloder, choose):
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0

        valid_entity = 0.0
        for i, batch in enumerate(dataloder):
            if choose:
                predictions = trainer.source_predict(batch)
            else :
                predictions = trainer.target_predict(batch)
            for pred in predictions:
                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('.', end='')

        s_mrr = MRR / valid_entity
        s_ndcg_5 = NDCG_5 / valid_entity
        s_ndcg_10 = NDCG_10 / valid_entity
        s_hr_1 = HT_1 / valid_entity
        s_hr_5 = HT_5 / valid_entity
        s_hr_10 = HT_10 / valid_entity


        return s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10


    s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = predict(source_valid_batch, 1)
    t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = predict(target_valid_batch, 0)

    print("\nsource: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))
    print("target: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))

    s_dev_score = s_mrr
    t_dev_score = t_mrr

    if s_dev_score > max(s_dev_score_history):
        print("source best!")
        s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = predict(source_test_batch, 1)
        print("\nsource: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))

    if t_dev_score > max(t_dev_score_history):
        print("target best!")
        t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = predict(target_test_batch, 0)
        print("target: \t{:.6f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))


    file_logger.log(
        "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, s_dev_score, max([s_dev_score] + s_dev_score_history)))

    print(
        "epoch {}: train_loss = {:.6f}, source_hit = {:.4f}, source_ndcg = {:.4f}, target_hit = {:.4f}, target_ndcg = {:.4f}".format(
            epoch, \
            train_loss, s_hr_10, s_ndcg_10, t_hr_10, t_ndcg_10))


    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    if epoch == 1 or s_dev_score > max(s_dev_score_history):
        # copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")

    # lr schedule
    if len(s_dev_score_history) > opt['decay_epoch'] and s_dev_score <= s_dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    s_dev_score_history += [s_dev_score]
    t_dev_score_history += [t_dev_score]
    print("")


"""
CUDA_VISIBLE_DEVICES=1 python -u train_rec.py --id gv --dataset game_video --model CDRIB --GNN 3 --beta 0.5 
"""