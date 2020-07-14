import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

import model
import metric
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
parser.add_argument("--num_epoch",type=int,default=1,help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int,default=20,help="predictive factors numbers in the model")
parser.add_argument("--num_ng", type=int,default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int,default=99, help="sample part of negative items for testing")
parser.add_argument("--out",default=True,help="save model or not")
args = parser.parse_args()

############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
print('user_num',user_num)
print('item_num',item_num)

# construct the train and test datasets
train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False)

########################### CREATE MODEL #################################
model = model.BPRmodel(user_num, item_num, args.factor_num)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)
# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
loss_list=[]
for epoch in range(args.num_epoch):
	model.train() 
	#start_time = time.time()
	train_loader.dataset.ng_sample()
	for user, item_i, item_j in train_loader:        #在一个epoch中每次训练batch个数据
		model.zero_grad()
		prediction_i, prediction_j = model(user, item_i, item_j)
		#print('prediction_i: ',prediction_i.size())
		loss = - (prediction_i - prediction_j).sigmoid().log().sum()
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		#count += 1
	loss_list.append(loss)
	print("Epoch: {}, loss: {}".format(epoch + 1, round(loss.item(), 5)))

	model.eval()
	F1 = metric.f_score(model, test_loader, args.top_k)
	print("F1: ",F1)

plt.plot(loss_list)
plt.title('Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()












