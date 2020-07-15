import torch
import torch.nn as nn
import numpy as np

import dataset

#rmse
def RMSE(pred_rate,real_rate):
    #使用均方根误差作为评价指标
    loss_func=nn.MSELoss()
    mse_loss=loss_func(pred_rate,real_rate)
    rmse_loss=torch.sqrt(mse_loss)
    return rmse_loss

#某个用户u的p，r计算
def precision_recall(recommend_u,test_u):
    p,r=0,0
    #print('recommend_u',set(recommend_u))
    #print('test_u',set(test_u))
    intersection=list(set(recommend_u).intersection(set(test_u)))
    #print('intersection:',intersection)
    val=len(intersection)
    #print('len of intersection:',val)
    p = val / len(recommend_u)
    r = val / len(test_u)
    return p,r

#f1-score
def f_score(model, test_loader, top_k):
    result = []
    test_u = []

    for user,item_i,item_j in test_loader:

        test_u.clear()
        test_u.append(item_i[0].item())
        #print('test_u:',test_u)

        prediction_i, prediction_j = model(user, item_i, item_j)  # 测试时（test_data），此处的i j是一样的
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        # 求tensor中某个dim的前k大或者前k小的值以及对应的index。  返回values,indices
        _, indices = torch.topk(prediction_i, top_k)

        recommend_u = torch.take(item_i, indices).numpy().tolist()
        #recommend_u=[25,133,207,208,222,396,74,91,514,659,820]

        p,r=precision_recall(recommend_u,test_u)
        if p==0 or r==0:
            f1=0
        else:
            f1 = round(2 * p * r / (p + r), 6)
        #if p!=0 :
            #print('u_p:',p)
            #print('u_r:',r)
            #print('u_f1:',f1)
        result.append(f1)
    return np.mean(result)

def f_score_v2(model, testdata, top_k,test_user_ratings):
    result = []
    test_u = []
    for user,items in testdata.key():
        test_u=test_user_ratings[user]
        prediction_i, prediction_j = model(user, items, items)
        _, indices = torch.topk(prediction_i, top_k)
        recommend_u = torch.take(items, indices).numpy().tolist()
        p,r=precision_recall(recommend_u,test_u)
        if p==0 or r==0:
            f1=0
        else:
            f1 = round(2 * p * r / (p + r), 6)
        result.append(f1)
    return np.mean(result)


def f_score_v3(model, test_loader, top_k):
    result = []
    test_u = []

    test_user_ratings=dataset.load_test_data()

    for user, item_i, item_j in test_loader:

        test_u.clear()
        if user[0].item() in test_user_ratings.keys():
            test_u=test_user_ratings[user[0].item()]
            test_u=list(test_u)
        test_u.append(item_i[0].item())


        prediction_i, prediction_j = model(user, item_i, item_j)  # 测试时（test_data），此处的i j是一样的
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        # 求tensor中某个dim的前k大或者前k小的值以及对应的index。  返回values,indices
        _, indices = torch.topk(prediction_i, top_k)

        recommend_u = torch.take(item_i, indices).numpy().tolist()
        # recommend_u=[25,133,207,208,222,396,74,91,514,659,820]

        p, r = precision_recall(recommend_u, test_u)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = round(2 * p * r / (p + r), 6)

        if len(test_u) >1:
            print('test_u:',test_u)
            print('len of test_u:',len(test_u))
            print('u_p:',p)
            print('u_r:',r)
            print('u_f1:',f1)

        result.append(f1)
    return np.mean(result)