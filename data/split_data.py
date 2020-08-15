import pandas as pd
import numpy as np
import random
from sklearn import model_selection


#随机打乱顺序 or 按时间排序
def split_basic(path='./dataset/ratings.csv',per=0.8,by_time=False):
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)  # 按照列值排序
        data.to_csv('dataset/TimeAscend.csv', index=False)  # 把新的数据写入文件
        trainpath="dataset/train_time.csv"
        testpath="dataset/test_time.csv"
    else:
        trainpath="dataset/train_shuffle.csv"
        testpath="dataset/test_shuffle.csv"

    traindata,testdata = model_selection.train_test_split(data, train_size=per, shuffle=not by_time)     # cross validation

    traindata.to_csv(trainpath, header=None, index=False)
    testdata.to_csv(testpath, header=None, index=False)
    print('dataset_num:',len(data))
    print('trainset_num:',len(traindata))
    print('testset_num:',len(testdata))

#按用户 取比例
def split_user_ratio(path='./dataset/ratings.csv',per=0.8,by_time=False):
    trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)
        data.to_csv('dataset/TimeAscend.csv', index=False)

    user=data['user'].unique()
    for u in user:
        u_data=data[data['user'].isin([u])]
        u_traindata, u_testdata = model_selection.train_test_split(u_data, train_size=per, shuffle=not by_time)
        trainset=trainset.append(u_traindata)
        testset=testset.append(u_testdata)

    trainset.to_csv("dataset/train_user_retio.csv", header=None, index=False)
    testset.to_csv("dataset/test_user_retio.csv", header=None, index=False)
    print('dataset_num:',len(data))
    print('trainset_num:',len(trainset))
    print('testset_num:',len(testset))

#按用户 取个数
def split_user_num(path='./dataset/ratings.csv',u_test_num=2,by_time=False):
    trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)
        data.to_csv('dataset/TimeAscend.csv', index=False)

    user=data['user'].unique()
    for u in user:
        u_data=data[data['user'].isin([u])]
        u_traindata, u_testdata = model_selection.train_test_split(u_data, test_size=u_test_num, shuffle=not by_time)
        trainset=trainset.append(u_traindata)
        testset=testset.append(u_testdata)

    trainset.to_csv("dataset/train_user_num.csv", header=None, index=False)
    testset.to_csv("dataset/test_user_num.csv", header=None, index=False)
    print('dataset_num:',len(data))
    print('trainset_num:',len(trainset))
    print('testset_num:',len(testset))



#自己实现部分
def split_basic_v2(path='./dataset/ratings.csv',per=0.8,by_time=False):
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)
        data.to_csv('dataset/TimeAscendV2.csv', index=False)
        data = np.array(data)
        trainpath="dataset/train_time2.csv"
        testpath="dataset/test_time2.csv"
    else:
        #随机打乱顺序的的三种方法
        #data = data.sample(frac=1)
        #data.to_csv('data/shuffle.csv', index=False)
        data = np.array(data)
        #random.shuffle(data)
        data = np.random.permutation(data)
        trainpath="dataset/train_shuffle2.csv"
        testpath="dataset/test_shuffle2.csv"

    # 取前80%为训练集
    total_num = [d[0] for d in data]
    print('total_num:',len(total_num))
    trainset = data[:int(per * len(total_num))]
    print('trainset_num:',len(trainset))
    trainset = pd.DataFrame(trainset,columns=['user', 'item','rate','time'])
    trainset[['user','item','time']]=trainset[['user','item','time']].astype('int')
    trainset.to_csv(trainpath, header=None, index=False)

    # 剩余20%为测试集
    testset = data[int(per * len(total_num)):]
    print('testset_num:',len(testset))
    testset = pd.DataFrame(testset,columns=['user', 'item','rate','time'])
    testset[['user','item','time']]=testset[['user','item','time']].astype('int')
    testset.to_csv(testpath, header=None, index=False)

def split_user_ratio_v2(path='./dataset/ratings.csv',per=0.8,by_time=False):
    trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)
        data.to_csv('dataset/TimeAscendV2.csv', index=False)

    user=data['user'].unique()
    for u in user:
        u_data=data[data['user'].isin([u])]
        u_data = np.array(u_data)
        if not by_time:
            u_data = np.random.permutation(u_data)
        total_num = [d[0] for d in u_data]
        u_traindata = u_data[:int(per * len(total_num))]
        u_traindata = pd.DataFrame(u_traindata,columns=['user', 'item','rate','time'])
        trainset=trainset.append(u_traindata)

        u_testdata = u_data[int(per * len(total_num)):]
        u_testdata = pd.DataFrame(u_testdata,columns=['user', 'item','rate','time'])
        testset=testset.append(u_testdata)

    trainset[['user','item','time']]=trainset[['user','item','time']].astype('int')
    testset[['user','item','time']]=testset[['user','item','time']].astype('int')
    trainset.to_csv("dataset/train_user_retio2.csv", header=None, index=False)
    testset.to_csv("dataset/test_user_retio2.csv", header=None, index=False)
    print('dataset_num:',len(data))
    print('trainset_num:',len(trainset))
    print('testset_num:',len(testset))

def split_user_num_v2(path='./dataset/ratings.csv',u_test_num=2,by_time=False):
    trainset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    testset = pd.DataFrame(columns=('user', 'item', 'rate', 'time'))
    data = pd.read_csv(path, header=None, names=['user', 'item','rate','time'], sep=',')

    if by_time:
        data = data.sort_values(by="time", ascending=True)
        data.to_csv('dataset/TimeAscendV2.csv', index=False)

    user=data['user'].unique()
    for u in user:
        u_data=data[data['user'].isin([u])]
        u_data = np.array(u_data)
        if not by_time:
            u_data = np.random.permutation(u_data)
        u_traindata = u_data[:-u_test_num]
        u_traindata = pd.DataFrame(u_traindata,columns=['user', 'item','rate','time'])
        trainset=trainset.append(u_traindata)

        u_testdata = u_data[-u_test_num:]
        u_testdata = pd.DataFrame(u_testdata,columns=['user', 'item','rate','time'])
        testset=testset.append(u_testdata)

    trainset[['user','item','time']]=trainset[['user','item','time']].astype('int')
    testset[['user','item','time']]=testset[['user','item','time']].astype('int')
    trainset.to_csv("dataset/train_user_num2.csv", header=None, index=False)
    testset.to_csv("dataset/test_user_num2.csv", header=None, index=False)
    print('dataset_num:',len(data))
    print('trainset_num:',len(trainset))
    print('testset_num:',len(testset))



#split_basic()












