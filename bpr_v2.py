import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import model
import metrics
import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
parser.add_argument("--num_epoch",type=int,default=2,help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int,default=20,help="predictive factors numbers in the model")
parser.add_argument("--num_ng", type=int,default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int,default=99, help="sample part of negative items for testing")
parser.add_argument("--out",default=True,help="save model or not")
args = parser.parse_args()

user_num, item_num, all_user_ratings, train_user_ratings, test_user_ratings=dataset.load_data()
traindata=dataset.generate_traindata(train_user_ratings,all_user_ratings,item_num)
testdata=dataset.generate_testdata(test_user_ratings,train_user_ratings,item_num)
model = model.BPRmodel(user_num, item_num, args.factor_num)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)

loss_list=[]
f1_list=[]
traindata =torch.Tensor(traindata)
for epoch in range(args.num_epoch):
    model.train()
    epoch_loss=[]
    for i in range(traindata.size()[0]):
        model.zero_grad()
        prediction_i, prediction_j = model(traindata[i][0], traindata[i][1],traindata[i][2])   #问题：TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not int
        loss = - (prediction_i - prediction_j).sigmoid().log()
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
    l=np.sum(epoch_loss)
    loss_list.append(l)
    print("Epoch: {}, loss: {}".format(epoch + 1, round(loss.item(), 5)))

    model.eval()
    F1 = metrics.f_score_v2(model, testdata, args.top_k, test_user_ratings)
    f1_list.append(F1)
    print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5), round(F1, 5)))

plt.plot(loss_list)
plt.title('Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# plt.savefig("/home/zhangjingsen/MF_BPR/loss.jpg")
plt.clf()  # 画完第一个图之后重置

plt.plot(f1_list)
plt.title('Testing')
plt.xlabel('Epochs')
plt.ylabel('F1-score')
plt.show()
# plt.savefig("/home/zhangjingsen/MF_BPR/f1.jpg")

# np.savetxt("/home/zhangjingsen/MF_BPR/loss.txt", loss_list)
# np.savetxt("/home/zhangjingsen/MF_BPR/f1.txt", f1_list)












