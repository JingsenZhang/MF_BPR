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

import model
import metric
import data_utils
import save
import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=int,default=1,help="1:MF  2:BPR")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
parser.add_argument("--num_epoch",type=int,default=1,help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_dim", type=int,default=20,help="predictive factors numbers in the model")
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

        # test data
        testData = pd.read_csv('data/ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
        userIdx2 = torch.Tensor(testData.user.values).long()
        itemIdx2 = torch.Tensor(testData.item.values).long()
        rates2 = torch.Tensor(testData.rate.values).float()

        user_num = max(userIdx) + 1        # id可为0
        item_num = max(itemIdx) + 1
        print('user_num',user_num.item())
        print('item_num',item_num.item())

        model = model.MFmodel(user_num, item_num, args.factor_dim)
        #save.load_model(model,"models/model_MF")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambd)   # 权重衰减（正则化）
        criterion = nn.MSELoss()
        evaluate = metric.Metrics()

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

            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

            # test
            model.eval()
            rates_y2 = model(userIdx2, itemIdx2, global_mean)
            rmse = evaluate.RMSE(rates_y2, rates2)
            rmse_list.append(rmse)
            print('Epoch: {}, loss: {}, Test RMSE: {}'.format(epoch + 1, round(loss.item(), 5), round(rmse.item(), 5)))

        #save.save_model(model,'MF')
        '''
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
        '''
        #np.savetxt("rmse/rmse25.txt", rmse_list)


    #BPR
    else:
        print('BPR')
        train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
        print('user_num',user_num)
        print('item_num',item_num)

        # traindataset \ testdataset
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False)
        test_user_ratings = dataset.load_test_data()

        model = model.BPRmodel(user_num, item_num, args.factor_dim)
        #save.load_model(model,"models/model_BPR")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)
        evaluate = metric.Metrics()

        loss_list=[]
        f1_list=[]
        test_u=[]
        user_f1=0.0

        for epoch in range(args.num_epoch):
            #train
            model.train()
            train_loader.dataset.ng_sample()
            for user, item_i, item_j in train_loader:          #在一个epoch中每次训练batch个数据
                model.zero_grad()
                prediction_i = model.predict(user, item_i)
                prediction_j = model.predict(user, item_j)
                loss = - (prediction_i - prediction_j).sigmoid().log().sum()
                loss.backward()
                optimizer.step()
                # writer.add_scalar('data/loss', loss.item(), count)
            loss_list.append(loss)

            #test
            model.eval()
            for user, item_i, item_j in test_loader:
                test_u.clear()
                if user[0].item() in test_user_ratings.keys():
                    test_u = test_user_ratings[user[0].item()]
                    test_u = list(test_u)
                test_u.append(item_i[0].item())

                prediction_i = model.predict(user, item_i)
                # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor) 求tensor中某个dim的前k大或者前k小的值以及对应的index。  返回values,indices
                _, indices = torch.topk(prediction_i, args.top_k)

                recommend_u = torch.take(item_i, indices).numpy().tolist()
                user_f1 = evaluate.f1_score(recommend_u,test_u)
                f1_list.append(user_f1)
            f1_mean=np.mean(f1_list)
            print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5),round(f1_mean,5)))

        #save.save_model(model, 'BPR')

        '''
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
        '''

        #first_path = 'f1/'
        #all_path = first_path + 'D{}.txt'.format(args.factor_dim)
        #np.savetxt(all_path, f1_list)









