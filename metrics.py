import torch
import torch.nn as nn
import numpy as np

import dataset


class Metric():

    def __init__(self):
        pass

    #rmse
    def RMSE(self,pred_rate,real_rate):
        #使用均方根误差作为评价指标
        loss_func=nn.MSELoss()
        mse_loss=loss_func(pred_rate,real_rate)
        rmse_loss=torch.sqrt(mse_loss)
        return rmse_loss

    #mse
    def MSE(self,pred_rate,real_rate):
        #使用均方误差作为评价指标
        loss_func=nn.MSELoss()
        mse_loss=loss_func(pred_rate,real_rate)
        return mse_loss

    #mae
    def MAE(self,pred_rate,real_rate):
        #使用平均绝对误差作为评价指标
        mae_loss=abs(pred_rate-real_rate).sum()/pred_rate.size(0)
        return mae_loss

    #某用户u的p，r计算
    def precision_recall(self,recommend_u,test_u):
        p,r=0,0
        intersection=list(set(recommend_u).intersection(set(test_u)))
        val=len(intersection)
        #print('len of intersection:',val)
        p = val / len(recommend_u)
        r = val / len(test_u)
        return p,r

    #mean
    def mean(self,result):
        return np.mean(result)

    #f1-score
    def f1_score(self,recommend_u, test_u):
        p, r = self.precision_recall(recommend_u, test_u)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = round(2 * p * r / (p + r), 6)
        return f1








    #最初写的f1-score
    def f1_score_v2(self,model, test_loader, top_k):
        result = []
        test_u = []
        test_user_ratings = dataset.load_test_data()

        for user,item_i,item_j in test_loader:
            test_u.clear()
            if user[0].item() in test_user_ratings.keys():
                test_u = test_user_ratings[user[0].item()]
                test_u = list(test_u)
            test_u.append(item_i[0].item())

            prediction_i, prediction_j = model(user, item_i, item_j)  # 测试时（test_data），此处的i j是一样的
            # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)求tensor中某个dim的前k大或者前k小的值以及对应的index。  返回values,indices
            _, indices = torch.topk(prediction_i, top_k)
            recommend_u = torch.take(item_i, indices).numpy().tolist()
            p,r=self.precision_recall(recommend_u,test_u)
            if p==0 or r==0:
                f1=0
            else:
                f1 = round(2 * p * r / (p + r), 6)
            result.append(f1)
        return np.mean(result)

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