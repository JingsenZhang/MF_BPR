import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import metric
import numpy as np
import sys

import model

parser = argparse.ArgumentParser(description='MF')
parser.add_argument('--factor_num', default=20, type=int, help='latent dimension')
parser.add_argument('--lambd', default=1e-5, type=float, help='regularization penalty')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--num_epoch', default=2, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    #train data
    trainData = pd.read_csv('data/ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    userIdx = trainData.user.values
    itemIdx = trainData.item.values
    rates = trainData.rate.values
    userIdx = torch.Tensor(trainData.user.values).long()
    itemIdx = torch.Tensor(trainData.item.values).long()
    rates = torch.Tensor(trainData.rate.values).float()
    rate=rates.numpy()
    global_mean=np.mean(rate)
    #print('global_mean:',global_mean)

    #test data
    testData = pd.read_csv('data/ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    userIdx2 = torch.Tensor(testData.user.values).long()
    itemIdx2 = torch.Tensor(testData.item.values).long()
    rates2 = torch.Tensor(testData.rate.values).float()

    user_num = max(userIdx) + 1   #id可为0
    item_num = max(itemIdx) + 1
    #print(user_num, item_num)

    mf = model.MFmodel(user_num, item_num, args.factor_num)
    optimizer = optim.Adam(mf.parameters(), lr=args.lr, weight_decay=args.lambd)    #权重衰减（正则化）
    criterion = nn.MSELoss()

    loss_list=[]
    rmse_list=[]

    for i in range(int(args.num_epoch)):
        #train
        mf.train()
        optimizer.zero_grad()
        rates_y = mf(userIdx, itemIdx, global_mean)
        loss = criterion(rates_y, rates)
        loss.backward()
        optimizer.step()
        loss_list.append(loss)

        # test
        mf.eval()
        rates_y2 = mf(userIdx2, itemIdx2, global_mean)
        rmse = metric.RMSE(rates_y2, rates2)
        rmse_list.append(rmse)

        print('Epoch: {}, loss: {}, Test RMSE: {}'.format(i + 1, round(loss.item(), 5),round(rmse.item(),5)))

    plt.figure(1)
    plt.plot(loss_list)
    plt.title('Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(2)
    plt.plot(rmse_list)
    plt.title('Testing')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.show()