import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from collections import defaultdict
import torch.utils.data as data


def load_all(train_rating_path='dataset/train_user_num.csv',
			 test_rating_path='dataset/test_user_num.csv',
			 test_samples_num=100,
			 test_negative='dataset/ml-1m.test.negative'):
	""" We load all the file here to save time in each epoch. """

	# 用于训练的用户真实评分   [[u1,i1],[u2,i2]...]    训练集：有过行为的用户和物品组
	train_user_rating = pd.read_csv(train_rating_path, sep=',', header=None, names=['user', 'item'], usecols=[0, 1],
									dtype={0: np.int32, 1: np.int32})
	user_num = train_user_rating['user'].max() + 1
	item_num = train_user_rating['item'].max() + 1
	print('user_num', user_num)
	print('item_num', item_num)
	user_list = train_user_rating['user'].unique()

	train_user_rating = train_user_rating.values.tolist()

	# load ratings as a matrix  索引从0开始
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	for x in train_user_rating:
		train_mat[x[0], x[1]] = 1.0  # 由训练集生成的用户与物品矩阵，已经有过行为的值为1

	# 用于测试的用户真实评分  test_user_ratings:  defaultdict(<type 'set()'>, { 'u1': {i1,i2....}, 'u2': {i5.i6....} } )
	test_user_ratings = defaultdict(set)
	with open(test_rating_path, 'r') as f:
		for line in f.readlines():
			u, i, _, _ = line.split(',')
			u = int(u)
			i = int(i)
			test_user_ratings[u].add(i)

	#候选列表（指定大小test_samples_num）
	#使用数据集test_user_num，凑齐100个（2+98）
	candidate_list = []
	for u in user_list:
		for i in test_user_ratings[u]:
			candidate_list.append([u, i])
		for t in range(2,test_samples_num):
			i = np.random.randint(item_num)
			while (u, i) in train_mat:
				i = np.random.randint(item_num)
			candidate_list.append([u, i])

	'''
	#使用原数据集test_negative，凑齐100个（1+99）
	candidate_list = []
	with open(test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			u = eval(arr[0])[0]
			candidate_list.append([u, eval(arr[0])[1]])
			for i in arr[1:]:
				test_data.append([u, int(i)])             # [[0,25],[0,1064],[0,2791]...[1,133],[1,1072]...]  测试集
			line = fd.readline()
	'''

	'''
	#随机抽取不在metrix中的100个
	candidate_list=[]
	for u in user_list:
		for t in range(test_samples_num):
			i = np.random.randint(item_num)
			while (u, i) in train_mat:
				i = np.random.randint(item_num)
			candidate_list.append([u, i])
	'''

	return train_user_rating, test_user_ratings, candidate_list,user_num, item_num, train_mat, user_list

#生成某个用户的候选列表（all ranking）
def generate_candidate_u(u, item_num, train_mat):
	candidate_u_list = []
	for i in range(1, item_num):
		if not ((u, i) in train_mat):
			candidate_u_list.append([u, i])
	return candidate_u_list


class BPRData(data.Dataset):

	def __init__(self, data, num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		self.data = data
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def negative_sampling(self):
		assert self.is_training, 'no need to sampling when testing'
		self.data_sample = []
		for x in self.data:                                #data为train_data
			u, i = x[0], x[1]                              #正样本i
			for t in range(self.num_ng):                   #每个用户的一个（u，i）都选取num_ng个负样本j,控制正负样本比例ratio
				j = np.random.randint(self.num_item)       #随机选取一个不在用户物品表（未产生过行为）的物品j作为负样本
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.data_sample.append([u, i, j])         #三元组训练集[u,i,j]

	def __len__(self):
		return self.num_ng * len(self.data) if self.is_training else len(self.data)

	def __getitem__(self, idx):
		data = self.data_sample if self.is_training else self.data    #训练集取三元组，测试集取二元组

		user = data[idx][0]
		item_i = data[idx][1]
		item_j = data[idx][2] if self.is_training else data[idx][1]   #统一成三元组
		return user, item_i, item_j














