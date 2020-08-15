import time
import sys
import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import evaluate
from util import utils
from util.logger import Logger
from data import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default='MF',help="MF BPR GMF MLP NeuMF")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument('--dropout',type=float,default=0.0,help='dropout rate')
parser.add_argument("--lambd",type=float, default=0.001, help="model regularization rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--num_epoch",type=int,default=2,help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")

parser.add_argument("--factor_dim", type=int,default=20,help="predictive factors numbers in the model")
parser.add_argument('--factor_dim_GMF',type=int,default=32,help='in NeuMF,dimension of embedding in GMF submodel')
parser.add_argument('--factor_dim_MLP',type=int,default=128,help='in NeuMF,dimension of embedding in MLP submodel')
parser.add_argument('--hidden_layer_MLP',type=list,default=[128,64,32],help='hidden layers in MLP')
parser.add_argument('--pre_training',type=bool,default=False,help='use pre-training')

parser.add_argument("--num_ng", type=int,default=4, help="sample negative items for training")
parser.add_argument("--test_samples_num", type=int,default=99, help="sample part of negative items for testing")
parser.add_argument("--out",type=bool,default=True,help="save model or not")
parser.add_argument('--model_path',type=str,default='./result/')
args = parser.parse_args()

args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark=True


if __name__=='__main__':
    '''
    #log
    timestamp = time.time()
    run_id = "%.8f" % (timestamp)
    log_dir='result/'+args.model+'_log/'+run_id+'.log'
    log_error_dir = 'result/' + args.model + '_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary
    '''
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

        model = model.MF(user_num, item_num, args.factor_dim)
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
            rmse = evaluate.RMSE(rates_y2, rates2)
            rmse_list.append(rmse)
            print('Epoch: {}, loss: {}, Test RMSE: {}'.format(epoch + 1, round(loss.item(), 5), round(rmse.item(), 5)))

        utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/mf_loss.jpg")
        utils.result_plot(rmse_list, 'Testing', 'Epochs', 'RMSE', "result/mf_rmse.jpg")
        #all_path = 'result/rmse/' + 'D{}.txt'.format(args.factor_dim)
        #utils.save_txt(all_path, rmse_list)
        #utils.save_model(model,'MF')


    elif args.model=='BPR':
        print('BPR')
        train_data, test_user_ratings, candidate_list, user_num ,item_num, train_mat,user_list = data_utils.load_all()
        # traindataset \ testdataset
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.BPRData(candidate_list, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset,batch_size=args.test_samples_num+1, shuffle=False)

        model = model.BPR(user_num, item_num, args.factor_dim)
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
                _, indices = torch.topk(prediction_i, args.top_k)
                recommend_u = torch.take(item_i, indices).numpy().tolist()

                user_f1 = evaluate.f1_score(recommend_u,test_u)
                f1_epoch_list.append(user_f1)
            f1_mean=np.mean(f1_epoch_list)
            f1_list.append(f1_mean)
            print("Epoch: {}, loss: {}, Test F1: {}".format(epoch + 1, round(loss.item(), 5),round(f1_mean,5)))

        utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/bpr_loss.jpg")
        utils.result_plot(f1_list, 'Testing', 'Epochs', 'F1-score', "result/bpr_f1.jpg")
        #all_path = 'result/f1/' + 'D{}.txt'.format(args.factor_dim)
        #utils.save_txt(all_path, f1_list)
        #utils.save_model(model, 'BPR')

        '''
        # BPR （all ranking情形）
        train_data, test_user_ratings, user_num ,item_num, train_mat ,user_list= data_utils.load_all(test_samples_num=args.test_samples_num)
        # traindataset
        train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)

        model = model.BPR(user_num, item_num, args.factor_dim)
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

        utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/bpr_loss.jpg")
        utils.result_plot(f1_list, 'Testing', 'Epochs', 'F1-score', "result/bpr_f1.jpg")
        '''

    elif args.model=='NeuMF':
        print('NeuMF')
        train_data, test_data, candidate_list, user_num, item_num, train_mat, user_list=data_utils.load_all()
        train_dataset = data_utils.NCFData(train_data,item_num,train_mat,args.num_ng,True)
        test_dataset = data_utils.NCFData(candidate_list, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.test_samples_num+1, shuffle=False,num_workers=0)

        GMF_model_path = os.path.join(args.model_path, 'GMF.pth')
        MLP_model_path = os.path.join(args.model_path, 'MLP.pth')
        if args.pre_training:
            assert os.path.exists(GMF_model_path), 'lack of GMF model'
            assert os.path.exists(MLP_model_path), 'lack of MLP model'
            GMF_model=torch.load(GMF_model_path)
            MLP_model=torch.load(MLP_model_path)
        else:
            GMF_model=None
            MLP_model=None
        model=model.NeuMF(user_num,item_num,args.factor_dim_GMF,args.factor_dim_MLP,args.hidden_layer_MLP,
                        args.dropout,args.pre_training,GMF_model,MLP_model)
        model.to(device=args.device)
        if args.pre_training:
            optimizer=optim.SGD(model.parameters(),lr=args.lr)
        else:
            optimizer=optim.Adam(model.parameters(),lr=args.lr)

        loss_list=[]
        HR_list=[]
        NDCG_list=[]

        for epoch in range(args.num_epoch):
            #train
            model.train()
            train_loader.dataset.negative_sampling()
            for user, item, label in train_loader:
                user=user.to(device=args.device)
                item=user.to(device=args.device)
                label=label.float().to(device=args.device)

                model.zero_grad()
                loss=model(user,item,label)
                loss.backward()
                optimizer.step()
            loss_list.append(loss)

            #test
            model.eval()
            HR,NDCG=[],[]
            for user,item,label in test_loader:
                user=user.to(device=args.device)
                item=item.to(device=args.device)
                prediction=model.predict(user,item)
                _,indices=torch.topk(prediction,args.top_k)
                recommend_u=torch.take(item,indices).cpu().numpy().tolist()

                gt_item=item[0].item()
                HR.append(evaluate.hit(gt_item,recommend_u))
                NDCG.append(evaluate.ndcg(gt_item,recommend_u))
            hr=np.mean(HR)
            ndcg=np.mean(NDCG)
            HR_list.append(hr)
            NDCG_list.append(ndcg)
            print("Epoch: {}, loss: {}, HR: {}, NDCG: {}".format(epoch + 1, round(loss.item(), 5), round(hr, 5),round(ndcg,5)))

        if args.out:
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model, os.path.join(args.model_path, 'NeuMF.pth'))

    elif args.model=='GMF':
        print('GMF')
        train_data, test_data, candidate_list, user_num, item_num, train_mat, user_list = data_utils.load_all()
        train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.NCFData(candidate_list, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.test_samples_num + 1, shuffle=False, num_workers=0)

        model = model.GMF(user_num, item_num, args.factor_dim_GMF,args.dropout)
        model.to(device=args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_list = []
        HR_list = []
        NDCG_list = []

        for epoch in range(args.num_epoch):
            # train
            model.train()
            train_loader.dataset.negative_sampling()
            for user, item, label in train_loader:
                user = user.to(device=args.device)
                item = user.to(device=args.device)
                label = label.float().to(device=args.device)

                model.zero_grad()
                loss = model(user, item, label)
                loss.backward()
                optimizer.step()
            loss_list.append(loss)

            # test
            model.eval()
            HR, NDCG = [], []
            for user, item, label in test_loader:
                user = user.to(device=args.device)
                item = item.to(device=args.device)
                prediction = model.predict(user, item)
                _, indices = torch.topk(prediction, args.top_k)
                recommend_u = torch.take(item, indices).cpu().numpy().tolist()

                gt_item = item[0].item()
                HR.append(evaluate.hit(gt_item, recommend_u))
                NDCG.append(evaluate.ndcg(gt_item, recommend_u))
            hr = np.mean(HR)
            ndcg = np.mean(NDCG)
            HR_list.append(hr)
            NDCG_list.append(ndcg)
            print("Epoch: {}, loss: {}, HR: {}, NDCG: {}".format(epoch + 1, round(loss.item(), 5), round(hr, 5),
                                                                 round(ndcg, 5)))
        if args.out:
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model, os.path.join(args.model_path, 'MLP.pth'))

    elif args.model=='MLP':
        print('MLP')
        train_data, test_data, candidate_list, user_num, item_num, train_mat, user_list = data_utils.load_all()
        train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
        test_dataset = data_utils.NCFData(candidate_list, item_num, train_mat, 0, False)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.test_samples_num + 1, shuffle=False, num_workers=0)

        model = model.MLP(user_num, item_num, args.factor_dim_MLP, args.hidden_layer_MLP, args.dropout )
        model.to(device=args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_list = []
        HR_list = []
        NDCG_list = []

        for epoch in range(args.num_epoch):
            # train
            model.train()
            train_loader.dataset.negative_sampling()
            for user, item, label in train_loader:
                user = user.to(device=args.device)
                item = user.to(device=args.device)
                label = label.float().to(device=args.device)

                model.zero_grad()
                loss = model(user, item, label)
                loss.backward()
                optimizer.step()
            loss_list.append(loss)

            # test
            model.eval()
            HR, NDCG = [], []
            for user, item, label in test_loader:
                user = user.to(device=args.device)
                item = item.to(device=args.device)
                prediction = model.predict(user, item)
                _, indices = torch.topk(prediction, args.top_k)
                recommend_u = torch.take(item, indices).cpu().numpy().tolist()

                gt_item = item[0].item()
                HR.append(evaluate.hit(gt_item, recommend_u))
                NDCG.append(evaluate.ndcg(gt_item, recommend_u))
            hr = np.mean(HR)
            ndcg = np.mean(NDCG)
            HR_list.append(hr)
            NDCG_list.append(ndcg)
            print("Epoch: {}, loss: {}, HR: {}, NDCG: {}".format(epoch + 1, round(loss.item(), 5), round(hr, 5), round(ndcg, 5)))

        if args.out:
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model, os.path.join(args.model_path, 'GMF.pth'))









