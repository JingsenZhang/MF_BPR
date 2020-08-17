import torch
import torch.nn as nn
import numpy as np


#rmse
def RMSE(pred_rate,real_rate):                        #若添加self，需定义类的实例调用
    #使用均方根误差作为评价指标
    loss_func=nn.MSELoss()
    mse_loss=loss_func(pred_rate,real_rate)
    rmse_loss=torch.sqrt(mse_loss)
    return rmse_loss

#mse
def MSE(pred_rate,real_rate):
    #使用均方误差作为评价指标
    loss_func=nn.MSELoss()
    mse_loss=loss_func(pred_rate,real_rate)
    return mse_loss

#mae
def MAE(pred_rate,real_rate):
    #使用平均绝对误差作为评价指标
    mae_loss=abs(pred_rate-real_rate).sum()/pred_rate.size(0)
    return mae_loss

#某用户u的p，r计算
def precision_recall(recommend_u,test_u):
    p,r=0,0
    intersection=list(set(recommend_u).intersection(set(test_u)))
    val=len(intersection)
    #print('len of intersection:',val)
    p = val / len(recommend_u)
    r = val / len(test_u)
    return p,r


#f1-score
def f1_score(recommend_u, test_u):
    p, r = precision_recall(recommend_u, test_u)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = round(2 * p * r / (p + r), 6)
    #print('p:', p)
    #print('r:', r)
    #print('f1:',f1)
    return f1

#HitRatio
def hit(gt_item,ranklist):
    if gt_item in ranklist:
        return 1
    return  0

#NDCG
def ndcg(gt_item,ranklist):
    if gt_item in ranklist:
        index=ranklist.index(gt_item)
        return np.log(2) / np.log(index+2)
    return 0











'''
调包
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
# 调整后的R square
def adj_r_squared(x_test,y_test,y_predict):
    SS_R = sum((y_test-y_predict)**2)
    SS_T = sum((y_test-np.mean(y_test))**2)
    r_squared = 1 - (float(SS_R))/SS_T
    adj_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
    return adj_r_squared
from scipy.stats import pearsonr   # 皮尔逊相关系数

#调用
mean_squared_error(y_test,y_predict)  #y_test,y_predict 为array
mean_absolute_error(y_test,y_predict)
r2_score(y_test,y_predict)
adj_r_squared(x_test,y_test,y_predict)
pearsonr(y_test,y_predict)
'''