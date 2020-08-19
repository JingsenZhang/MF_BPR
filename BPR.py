import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter

from data import data_utils
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
parser.add_argument("--embedding_dim", type=int, default=8, help="dimension of embedding")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="./dataset/")
parser.add_argument("--model_path", type=str, default="./result/")
parser.add_argument("--out", type=bool,default=True, help="save model or not")
parser.add_argument("--disable_cuda", action='store_true', help="Disable CUDA")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
cudnn.benchmark = True

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_dim):
        super(BPR, self).__init__()

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self,user,item_i,item_j):
        prediction_i = self.predict(user,item_i)
        prediction_j = self.predict(user,item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        return loss

    def predict(self,user,item):                #参数可为数值或list
        user = self.user_emb(user)
        item = self.item_emb(item)
        prediction = (user * item).sum(dim=-1)
        return prediction

if __name__=='__main__':
    print('BPR')

    #log
    timestamp = time.time()
    run_id = "%.2f" % (timestamp)
    log_dir='result/BPR_log/'+run_id+'.log'
    log_error_dir = 'result/BPR_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary

    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_user_ratings, candidate_list, user_num ,item_num, train_mat,user_list = data_utils.load_all(data_file)
    # traindataset \ testdataset

    train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.BPRData(candidate_list, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False)

    model = BPR(user_num, item_num, args.embedding_dim)
    #utils.load_model(model,"result/model_BPR")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambd)

    loss_list=[]
    #f1_list=[]
    #f1_epoch_list=[]
    #test_u=[]
    #user_f1=0.0
    count, best_hr = 0, 0
    HR, NDCG = [], []
    HR_list = []
    NDCG_list = []

    #train_loader.dataset.negative_sampling()
    for epoch in range(args.epochs):
        #train
        model.train()
        train_loader.dataset.ng_sample()
        for user, item_i, item_j in train_loader:          #在一个epoch中每次训练batch个数据
            model.zero_grad()
            loss = model(user,item_i,item_j)
            loss.backward()
            optimizer.step()
        loss_list.append(loss)

        #test
        model.eval()
        HR.clear()
        NDCG.clear()
        #f1_epoch_list.clear()
        hr, ndcg = 0, 0
        for user, item_i, item_j in test_loader:
            '''
            test_u.clear()
            if user[0].item() in test_user_ratings.keys():
                test_u = test_user_ratings[user[0].item()]
                test_u = list(test_u)
            test_u.append(item_i[0].item())
            '''
            prediction_i = model.predict(user, item_i)
            _, indices = torch.topk(prediction_i, args.top_k)
            recommend_u = torch.take(item_i, indices).cpu().numpy().tolist()

            gt_item = item_i[0].item()
            HR.append(evaluate.hit(gt_item, recommend_u))
            NDCG.append(evaluate.ndcg(gt_item, recommend_u))
            #f1_epoch_list.append(evaluate.f1_score(recommend_u, test_u))
        hr = np.mean(HR)
        ndcg = np.mean(NDCG)
        #f1_mean = np.mean(f1_epoch_list)
        HR_list.append(hr)
        NDCG_list.append(ndcg)
        #f1_list.append(f1_mean)
        print("BPR Epoch: {}, loss: {}, HR: {:.3f}, NDCG: {:.3f}".format(epoch + 1, round(loss.item(), 5), hr, ndcg))

        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch + 1
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, os.path.join(args.model_path, 'BPR.pth'))

    utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/BPR_loss.jpg")
    utils.result_plot(HR_list, 'Testing', 'Epochs', 'HR', "result/BPR_HR.jpg")
    utils.result_plot(NDCG_list, 'Testing', 'Epochs', 'NDCG', "result/BPR_NDCG.jpg")
    print("BPR End")
    print('best_hr:{:.3f}'.format(best_hr))
    print('best_ndcg:{:.3f}'.format(best_ndcg))


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
