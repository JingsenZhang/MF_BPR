import pandas as pd
import numpy as np
import random
from sklearn import model_selection

class Split_data:
    def __init__(self):
        pass

    def split_shuffle(path='./data/ratings.csv',per=0.8):
        data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

        traindata,testdata = model_selection.train_test_split(data, train_size=per, shuffle=True)     # cross validation

        traindata.to_csv("data/train_shuffle.csv", header=None, index=False)
        testdata.to_csv("data/test_shuffle.csv", header=None, index=False)
        print('dataset_num:',len(data))
        print('trainset_num:',len(traindata))
        print('testset_num:',len(testdata))

    def split_time(path='data/ratings.csv',per=0.8):
        data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

        data = data.sort_values(by="time", ascending=True)                        # 按照列值排序
        data.to_csv('data/TimeAscend.csv', index=False)                           # 把新的数据写入文件

        traindata,testdata = model_selection.train_test_split(data, train_size=per, shuffle=False)

        traindata.to_csv("data/train_time.csv", header=None, index=False)
        testdata.to_csv("data/test_time.csv", header=None, index=False)
        print('dataset_num:',len(data))
        print('trainset_num:',len(traindata))
        print('testset_num:',len(testdata))

    def split_user_ratio(path='./data/ratings.csv',per=0.8):
        trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
        testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))

        data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

        user=data['user'].unique()
        #print(user)
        for u in user:
            u_data=data[data['user'].isin([u])]
            u_traindata, u_testdata = model_selection.train_test_split(u_data, train_size=per, shuffle=True)
            #print(u_traindata)
            #print(u_testdata)
            trainset=trainset.append(u_traindata)
            testset=testset.append(u_testdata)

        trainset.to_csv("data/train_user_retio.csv", header=None, index=False)
        testset.to_csv("data/test_user_retio.csv", header=None, index=False)
        print('dataset_num:',len(data))
        print('trainset_num:',len(trainset))
        print('testset_num:',len(testset))

    def split_user_num(path='./data/ratings.csv',u_test_num=2):
        trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
        testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))

        data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

        user=data['user'].unique()
        for u in user:
            u_data=data[data['user'].isin([u])]
            u_traindata, u_testdata = model_selection.train_test_split(u_data, test_size=u_test_num, shuffle=True)
            trainset=trainset.append(u_traindata)
            testset=testset.append(u_testdata)

        trainset.to_csv("data/train_user_num.csv", header=None, index=False)
        testset.to_csv("data/test_user_num.csv", header=None, index=False)
        print('dataset_num:',len(data))
        print('trainset_num:',len(trainset))
        print('testset_num:',len(testset))

    def split_user_time(path='./data/ratings.csv', per=0.8):
        trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
        testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))

        data = pd.read_csv(path, header=None, names=['user', 'item', 'rate', 'time'], sep=',')
        data = data.sort_values(by="time", ascending=True)                        # 按照列值排序

        user = data['user'].unique()
        for u in user:
            u_data = data[data['user'].isin([u])]
            u_traindata, u_testdata = model_selection.train_test_split(u_data, train_size=per, shuffle=False)
            trainset = trainset.append(u_traindata)
            testset = testset.append(u_testdata)

        trainset.to_csv("data/train_user_time.csv", header=None, index=False)
        testset.to_csv("data/test_user_time.csv", header=None, index=False)
        print('dataset_num:', len(data))
        print('trainset_num:', len(trainset))
        print('testset_num:', len(testset))

    def split_shuffle_v2(path='./data/ratings.csv',per=0.8):
        data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

        #随机打乱顺序的的三种方法
        #data = data.sample(frac=1)
        #data.to_csv('data/shuffle.csv', index=False)
        data = np.array(data,dtype=int)
        #random.shuffle(data)
        data = np.random.permutation(data)

        # 取前80%为训练集
        total_num = [d[0] for d in data]
        print('total_num:',len(total_num))
        trainset = data[:int(per * len(total_num))]
        print('trainset_num:',len(trainset))
        trainset = pd.DataFrame(trainset)                               # 将np.array转为dataframe
        trainset.to_csv("data/train_shuffle2.csv", header=None, index=False)          # 写入csv

        # 剩余百分之20为测试集
        testset = data[int(per * len(total_num)):]
        print('testset_num:',len(testset))
        testset = pd.DataFrame(testset)
        testset.to_csv("data/test_shuffle2.csv", header=None, index=False)


Split_data.split_shuffle()