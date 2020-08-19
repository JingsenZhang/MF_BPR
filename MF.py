import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse
import numpy as np
import sys
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter

import evaluate
from util import utils
from util.logger import Logger

parser = argparse.ArgumentParser()


parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=2, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimension of embedding")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="./dataset/")
parser.add_argument("--model_path", type=str, default="./model")
parser.add_argument("--out", type=bool,default=True, help="save model or not")
parser.add_argument("--disable_cuda", action='store_true', help="Disable CUDA")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
cudnn.benchmark = True

class MF(nn.Module):    #BiasMF
    def __init__(self, user_num, item_num, factor_dim):
        super(MF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_dim = factor_dim

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        self.user_bias = nn.Embedding(user_num, 1)
        self.user_bias.weight.data = torch.zeros(self.user_num, 1).float()   #初始化偏置项
        self.item_bias = nn.Embedding(item_num, 1)
        self.item_bias.weight.data = torch.zeros(self.item_num, 1).float()

    def forward(self,user_indices, item_indeices, global_mean,rates):
        rates_y = self.predict(user_indices, item_indeices, global_mean)
        criterion = nn.MSELoss()
        loss = criterion(rates_y, rates)
        return loss

    def predict(self,user_indices, item_indeices, global_mean):
        user_vec = self.user_emb(user_indices)           #pu向量
        item_vec = self.item_emb(item_indeices)          #qi向量

        dot = torch.mul(user_vec, item_vec).sum(dim=1)   #在1维度上求和（即遍历隐类k）

        rates = dot + self.user_bias(user_indices).view(-1) + self.item_bias(item_indeices).view(-1) + global_mean    #BiasMF
        return rates


if __name__=='__main__':
    print('MF')

    #log
    timestamp = time.time()
    run_id = "%.2f" % (timestamp)
    log_dir='result/MF_log/'+run_id+'.log'
    log_error_dir = 'result/MF_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary

    # train data
    trainData = pd.read_csv('dataset/ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    userIdx = trainData.user.values
    itemIdx = trainData.item.values
    rates = trainData.rate.values
    userIdx = torch.Tensor(trainData.user.values).long()
    itemIdx = torch.Tensor(trainData.item.values).long()
    rates = torch.Tensor(trainData.rate.values).float()
    rate = rates.numpy()
    global_mean = np.mean(rate)

    # test data
    testData = pd.read_csv('dataset/ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    userIdx2 = torch.Tensor(testData.user.values).long()
    itemIdx2 = torch.Tensor(testData.item.values).long()
    rates2 = torch.Tensor(testData.rate.values).float()

    user_num = max(userIdx) + 1        # id可为0
    item_num = max(itemIdx) + 1
    print('user_num',user_num.item())
    print('item_num',item_num.item())

    model = MF(user_num, item_num, args.embedding_dim)
    #utils.load_model(model,"result/model_MF")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambd)   # 权重衰减（正则化）

    loss_list = []
    rmse_list = []
    curr_lr=args.lr

    for epoch in range(int(args.epochs)):
        # train
        model.train()
        optimizer.zero_grad()
        loss = model(userIdx, itemIdx, global_mean,rates)
        loss.backward()
        optimizer.step()
        loss_list.append(loss)

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            utils.update_lr(optimizer, curr_lr)

        # test
        model.eval()
        rates_y2 = model.predict(userIdx2, itemIdx2, global_mean)
        rmse = evaluate.RMSE(rates_y2, rates2)
        rmse_list.append(rmse)
        print('MF Epoch: {}, loss: {}, Test RMSE: {}'.format(epoch + 1, round(loss.item(), 5), round(rmse.item(), 5)))

    utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/mf_loss.jpg")
    utils.result_plot(rmse_list, 'Testing', 'Epochs', 'RMSE', "result/mf_rmse.jpg")
    #all_path = 'result/rmse/' + 'D{}.txt'.format(args.factor_dim)
    #utils.save_txt(all_path, rmse_list)
    #utils.save_model(model,'MF')
    print("MF End")

