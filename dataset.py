import numpy as np
import os
import random
from collections import defaultdict


def load_data(all_path='./data/ratings.csv',train_path='./data/train_samples.csv',test_path='./data/test_samples.csv'):
    all_user_ratings = defaultdict(set)
    train_user_ratings = defaultdict(set)
    test_user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1

    with open(all_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split(',')
            u = int(u)
            i = int(i)
            all_user_ratings[u].add(i)                            #all_user_ratings:  defaultdict(<type 'set()'>, { 'u1': {i1,i2....}, 'u2': {i5.i6....} } )
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print("max_u_id:", max_u_id)
    print("max_i_id:", max_i_id)

    with open(train_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split(',')
            u = int(u)
            i = int(i)
            train_user_ratings[u].add(i)                            #train_user_ratings:  defaultdict(<type 'set()'>, { 'u1': {i1,i2....}, 'u2': {i5.i6....} } )

    with open(test_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split(',')
            u = int(u)
            i = int(i)
            test_user_ratings[u].add(i)                            #test_user_ratings:  defaultdict(<type 'set()'>, { 'u1': {i1,i2....}, 'u2': {i5.i6....} } )

    return max_u_id, max_i_id, all_user_ratings, train_user_ratings, test_user_ratings

def generate_traindata(train_user_ratings, all_user_ratings, item_num):
    traindata = []
    for u in train_user_ratings.keys():
        i = random.sample(train_user_ratings[u], 1)[0]
        j = random.randint(1, item_num)
        while j in all_user_ratings[u]:
            j = random.randint(1, item_num)
        traindata.append([u, i, j])
    #return np.asarray(traindata)
    return traindata

def generate_testdata(test_user_ratings, train_user_ratings, item_num):
    no_user_ratings = defaultdict(set)
    for u in test_user_ratings.keys():
        j = random.randint(1, item_num)
        while j in train_user_ratings[u]:
            j = random.randint(1, item_num)
        no_user_ratings[u].add(j)
    return no_user_ratings


def load_test_data(test_path='./data/test_samples.csv'):
    test_user_ratings = defaultdict(set)

    with open(test_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split(',')
            u = int(u)
            i = int(i)
            test_user_ratings[u].add(i)                            #test_user_ratings:  defaultdict(<type 'set()'>, { 'u1': {i1,i2....}, 'u2': {i5.i6....} } )

    return test_user_ratings

