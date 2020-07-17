import torch
import torch.nn as nn


class MFmodel(nn.Module):
    def __init__(self, user_num, item_num, factor_dim):
        super(MFmodel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_dim = factor_dim

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        self.user_bias = nn.Embedding(user_num, 1)
        self.user_bias.weight.data = torch.zeros(self.user_num, 1).float()   #初始化偏置项
        self.item_bias = nn.Embedding(item_num, 1)
        self.item_bias.weight.data = torch.zeros(self.item_num, 1).float()

    def forward(self, user_indices, item_indeices, global_mean):
        user_vec = self.user_emb(user_indices)           #pu向量
        item_vec = self.item_emb(item_indeices)          #qi向量

        dot = torch.mul(user_vec, item_vec).sum(dim=1)   #在1维度上求和（即遍历隐类k）

        rates = dot + self.user_bias(user_indices).view(-1) + self.item_bias(item_indeices).view(-1) + global_mean    #BiasMF
        return rates

class BPRmodel(nn.Module):
    def __init__(self, user_num, item_num, factor_dim):
        super(BPRmodel, self).__init__()

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self):
        pass

    def predict(self,user,item):
        user = self.user_emb(user)
        item = self.item_emb(item)
        prediction = (user * item).sum(dim=-1)
        return prediction
