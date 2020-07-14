import pandas as pd
import numpy as np
import random

def split_data(path='./data/ratings.csv',per=0.8):
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')
    data = np.array(data,dtype=int)
    #random.shuffle(data)                                          # 随机打乱两种方法
    data = np.random.permutation(data)

    # 取前70%为训练集
    total_num = [d[0] for d in data]
    print(len(total_num))
    trainset = data[:int(per * len(total_num))]
    print(len(trainset))
    trainset = pd.DataFrame(trainset)                               # 将np.array转为dataframe
    trainset.to_csv("data/train_samples.csv", index=False)          # 写入csv

    # 剩余百分之30为测试集
    testset = data[int(per * len(total_num)):]
    print(len(testset))
    testset = pd.DataFrame(testset)
    testset.to_csv("data/test_samples.csv", index=False)

split_data()

