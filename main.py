import time
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.utils.data as data

import model
from util import utils
from util.logger import Logger
from evaluation import Metric
from data import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default='MF',help="1:MF  2:BPR")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
parser.add_argument("--num_epoch",type=int,default=1,help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_dim", type=int,default=20,help="predictive factors numbers in the model")
parser.add_argument("--num_ng", type=int,default=4, help="sample negative items for training")
parser.add_argument("--test_samples_num", type=int,default=99, help="sample part of negative items for testing")
parser.add_argument("--out",default=True,help="save model or not")
args = parser.parse_args()


if __name__=='__main__':

    #log
    timestamp = time.time()
    run_id = "%.8f" % (timestamp)
    log_dir='result/'+args.model+'_log/'+run_id+'.log'
    log_error_dir = 'result/' + args.model + '_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary

    #MF
    if args.model=='MF':
        print('MF')
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

        model = model.MFmodel(user_num, item_num, args.factor_dim)
        #utils.load_model(model,"result/model_MF")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambd)   # 权重衰减（正则化）

        loss_list = []
        rmse_list = []
        curr_lr=args.lr

        for epoch in range(int(args.num_epoch)):
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
            rmse = Metric.RMSE(rates_y2, rates2)
            rmse_list.append(rmse)
            print('Epoch: {}, loss: {}, Test RMSE: {}'.format(epoch + 1, round(loss.item(), 5), round(rmse.item(), 5)))

        #utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "image/mf_loss.jpg")
        #utils.result_plot(rmse_list, 'Testing', 'Epochs', 'RMSE', "image/mf_rmse.jpg")
        #all_path = 'result/rmse/' + 'D{}.txt'.format(args.factor_dim)
        #utils.save_txt(all_path, rmse_list)
        #utils.save_model(model,'MF')


    else:
        print('BPR')

        train_data, test_user_ratings, test_data, user_num ,item_num, train_mat,user_list = data_utils.load_all(test_samples_num=args.test_samples_num)

        # traindataset \ testdataset
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset,batch_size=args.test_samples_num, shuffle=False)

        model = model.BPRmodel(user_num, item_num, args.factor_dim)
        #utils.load_model(model,"result/model_BPR")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)

        loss_list=[]
        f1_list=[]
        f1_epoch_list=[]
        test_u=[]
        user_f1=0.0

        #train_loader.dataset.negative_sampling()
        for epoch in range(args.num_epoch):
            #train
            model.train()
            train_loader.dataset.negative_sampling()
            for user, item_i, item_j in train_loader:          #在一个epoch中每次训练batch个数据
                model.zero_grad()
                loss = model(user,item_i,item_j)
                loss.backward()
                optimizer.step()
            loss_list.append(loss)

            #test
            model.eval()
            f1_epoch_list.clear()
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

                user_f1 = Metric.f1_score(recommend_u,test_u)
                f1_epoch_list.append(user_f1)
            f1_mean=np.mean(f1_epoch_list)
            f1_list.append(f1_mean)
            print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5),round(f1_mean,5)))

        utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "image/bpr_loss.jpg")
        utils.result_plot(f1_list, 'Testing', 'Epochs', 'F1-score', "image/bpr_f1.jpg")
        #all_path = 'result/f1/' + 'D{}.txt'.format(args.factor_dim)
        #utils.save_txt(all_path, f1_list)
        #utils.save_model(model, 'BPR')



        '''
        # BPR  （all ranking情形）
        train_data, test_user_ratings, user_num ,item_num, train_mat ,user_list= data_utils.load_all(test_samples_num=args.test_samples_num)
        # traindataset
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)

        model = model.BPRmodel(user_num, item_num, args.factor_dim)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)

        loss_list=[]
        f1_list=[]
        f1_epoch_list=[]
        test_u=[]
        user_f1=0.0

        #train_loader.dataset.negative_sampling()
        for epoch in range(args.num_epoch):
            #train
            model.train()
            train_loader.dataset.negative_sampling()
            for user, item_i, item_j in train_loader:          #在一个epoch中每次训练batch个数据
                model.zero_grad()
                loss = model(user,item_i,item_j)
                loss.backward()
                optimizer.step()
            loss_list.append(loss)

            #test
            model.eval()
            f1_epoch_list.clear()
            for u in user_list:
                #print('u:',u)
                candidate_u_list=data_utils.generate_candidate_u(u,item_num,train_mat)
                u_list=[]
                i_list=[]
                for x in candidate_u_list:
                    u_list.append(x[0])
                    i_list.append(x[1])
                u_list = torch.tensor(u_list)
                i_list = torch.tensor(i_list)
                prediction_i = model.predict(u_list, i_list)
                _, indices = torch.topk(prediction_i, args.top_k)
                recommend_u = torch.take(i_list, indices).numpy().tolist()

                test_u.clear()
                if u in test_user_ratings.keys():
                    test_u = test_user_ratings[u]
                    test_u = list(test_u)
                test_u.append(i_list[0])

                user_f1 = Metric.f1_score(recommend_u,test_u)
                f1_epoch_list.append(user_f1)
                #print('user_f1:',user_f1)
            f1_mean=np.mean(f1_epoch_list)
            f1_list.append(f1_mean)
            print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5),round(f1_mean,5)))

        utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "image/bpr_loss.jpg")
        utils.result_plot(f1_list, 'Testing', 'Epochs', 'F1-score', "image/bpr_f1.jpg")
        '''









