import argparse
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

import model
import metric
import data_utils
import save

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=int,default=1,help="1:MF  2:BPR")
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

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    #MF
    if args.model==1:
        print('MF')
        # train data
        trainData = pd.read_csv('data/ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
        userIdx = trainData.user.values
        itemIdx = trainData.item.values
        rates = trainData.rate.values
        userIdx = torch.Tensor(trainData.user.values).long()
        itemIdx = torch.Tensor(trainData.item.values).long()
        rates = torch.Tensor(trainData.rate.values).float()
        rate = rates.numpy()
        global_mean = np.mean(rate)
        # print('global_mean:',global_mean)

        # test data
        testData = pd.read_csv('data/ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
        userIdx2 = torch.Tensor(testData.user.values).long()
        itemIdx2 = torch.Tensor(testData.item.values).long()
        rates2 = torch.Tensor(testData.rate.values).float()

        user_num = max(userIdx) + 1  # id可为0
        item_num = max(itemIdx) + 1
        print('user_num',user_num.item())
        print('item_num',item_num.item())

        model = model.MFmodel(user_num, item_num, args.factor_num)
        #save.load_model(model,"models/model_MF")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambd)  # 权重衰减（正则化）
        criterion = nn.MSELoss()

        loss_list = []
        rmse_list = []
        curr_lr=args.lr

        for epoch in range(int(args.num_epoch)):
            # train
            model.train()
            optimizer.zero_grad()
            rates_y = model(userIdx, itemIdx, global_mean)
            loss = criterion(rates_y, rates)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
            '''
            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)
            '''
            # test
            model.eval()
            rates_y2 = model(userIdx2, itemIdx2, global_mean)
            rmse = metric.RMSE(rates_y2, rates2)
            rmse_list.append(rmse)
            print('Epoch: {}, loss: {}, Test RMSE: {}'.format(epoch + 1, round(loss.item(), 5), round(rmse.item(), 5)))

        #save.save_model(model,'MF')

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

        np.savetxt("rmse/rmse25.txt", rmse_list)


    #BPR
    else:
        print('BPR')
        train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
        print('user_num',user_num)
        print('item_num',item_num)

        # construct the train and test datasets
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False)

        model = model.BPRmodel(user_num, item_num, args.factor_num)
        save.load_model(model,"models/model_BPR")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)
        # writer = SummaryWriter() # for visualization

        loss_list=[]
        f1_list=[]
        curr_lr=args.lr

        for epoch in range(args.num_epoch):
            model.train()
            train_loader.dataset.ng_sample()
            for user, item_i, item_j in train_loader:        #在一个epoch中每次训练batch个数据
                model.zero_grad()
                prediction_i, prediction_j = model(user, item_i, item_j)
                #print('prediction_i: ',prediction_i.size())
                loss = - (prediction_i - prediction_j).sigmoid().log().sum()
                loss.backward()
                optimizer.step()
                # writer.add_scalar('data/loss', loss.item(), count)
            loss_list.append(loss)

            '''
            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)
            '''

            model.eval()
            F1 = metric.f_score_v3(model, test_loader, args.top_k)
            f1_list.append(F1)
            print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5),round(F1,5)))

        #save.save_model(model, 'BPR')

        plt.plot(loss_list)
        plt.title('Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        #plt.savefig("/home/zhangjingsen/MF_BPR/loss.jpg")
        plt.clf()  #画完第一个图之后重置

        plt.plot(f1_list)
        plt.title('Testing')
        plt.xlabel('Epochs')
        plt.ylabel('F1-score')
        plt.show()
        #plt.savefig("/home/zhangjingsen/MF_BPR/f1.jpg")

        #np.savetxt("/home/zhangjingsen/MF_BPR/loss.txt", loss_list)
        #np.savetxt("/home/zhangjingsen/MF_BPR/f1.txt", f1_list)










