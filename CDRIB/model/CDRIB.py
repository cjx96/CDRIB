import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.VBGE import VBGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class CDRIB(nn.Module):
    def __init__(self, opt):
        super(CDRIB, self).__init__()
        self.opt=opt

        self.source_GNN = VBGE(opt)
        self.target_GNN = VBGE(opt)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.dis = nn.Bilinear(opt["feature_dim"], opt["feature_dim"], 1)
        self.discri = nn.Sequential(
            nn.Linear(opt["feature_dim"]*2 * opt["GNN"], opt["feature_dim"]),
            nn.ReLU(),
            nn.Linear(opt["feature_dim"], 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.dropout = opt["dropout"]

        # self.user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])

        # self.shared_user = torch.arange(0, self.opt["shared_user"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.criterion.cuda()
            # self.shared_user = self.shared_user.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def HingeLoss(self, pos, neg):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def dis(self, A, B):
        C = torch.cat((A,B), dim = 1)
        return self.discri(C)

    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_user, source_learn_item = self.source_GNN(source_user, source_item, source_UV, source_VU)
        target_learn_user, target_learn_item = self.target_GNN(target_user, target_item, target_UV, target_VU)

        if self.source_user_embedding.training:
            per_stable = torch.randperm(self.opt["shared_user"])[:self.opt["user_batch_size"]].cuda(source_learn_user.device)

            pos = self.dis(self.my_index_select(source_learn_user, per_stable),
                           self.my_index_select(target_learn_user, per_stable)).view(-1)
            per = torch.randperm(self.opt["target_user_num"])[:self.opt["user_batch_size"]].cuda(pos.device)
            neg_share = self.my_index_select(target_learn_user, per)
            neg_1 = self.dis(self.my_index_select(source_learn_user, per_stable), neg_share).view(-1)

            per = torch.randperm(self.opt["source_user_num"])[:self.opt["user_batch_size"]].cuda(pos.device)
            neg_share = self.my_index_select(source_learn_user, per)
            neg_2 = self.dis(neg_share, self.my_index_select(target_learn_user, per_stable)).view(-1)

            if self.opt['bce']:
                pos_label, neg_label = torch.ones(pos.size()), torch.zeros(
                    neg_1.size())

                if self.opt["cuda"]:
                    pos_label = pos_label.cuda()
                    neg_label = neg_label.cuda()

                self.critic_loss = self.criterion(pos, pos_label) + self.criterion(neg_1, neg_label) +self.criterion(neg_2, neg_label)
            else :
                self.critic_loss = self.HingeLoss(pos, neg_1) + self.HingeLoss(pos, neg_2)

            source_learn_user_concat = torch.cat((target_learn_user[:self.opt["shared_user"]], source_learn_user[self.opt["shared_user"]:]),dim=0)
            target_learn_user_concat = torch.cat((source_learn_user[:self.opt["shared_user"]], target_learn_user[self.opt["shared_user"]:]),dim=0)

        else :
            source_learn_user_concat = torch.cat((target_learn_user[:self.opt["source_shared_user"]], source_learn_user[self.opt["source_shared_user"]:]), dim=0)
            target_learn_user_concat = torch.cat((source_learn_user[:self.opt["target_shared_user"]], target_learn_user[self.opt["target_shared_user"]:]), dim=0)

        return source_learn_user_concat, source_learn_item, target_learn_user_concat, target_learn_item