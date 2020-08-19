import subprocess
import numpy as np
import sys

#执行cmd
def run(command):
    subprocess.call(command, shell=True)

#执行cmd,并返回结果
def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'best_hr:' in line:
            best_hr = line
        if 'best_ndcg:' in line:
            best_ndcg = line
    return best_hr, best_ndcg

#指定参数和cmd
def per_data(data_set):
    parameter='embedding_dim'
    best_HR,best_NDCG=0,0
    for embedding_dim in [8,16]:
        cmd = 'python BPR.py ' + \
              ' --embedding_dim ' + str(embedding_dim) + ' ' + \
              ' --data_set ' + data_set + ' ' + \
              ' --batch_size 256 '
        print(cmd)
        best_hr,best_ndcg = sh(cmd)
        print('parameter：{}  {}'.format(str(parameter),embedding_dim))
        print(best_hr)
        print(best_ndcg)
        if eval(best_hr.split(':')[1].strip()) > best_HR:
            best_embedding_dim = embedding_dim
            best_HR = eval(best_hr.split(':')[1].strip())
        sys.stdout.flush()

    print(' batch_size ')
    best_HR, best_NDCG = 0, 0
    for batch_size in [128,256]:
        cmd = 'python BPR.py ' + \
              ' --embedding_dim ' + str(best_embedding_dim) + ' ' + \
              ' --data_set ' + data_set + ' ' + \
              ' --batch_size ' + str(batch_size)
        print(cmd)
        best_hr,best_ndcg = sh(cmd)
        print('parameter：{}  {}'.format(str(parameter),batch_size))
        print(best_hr)
        print(best_ndcg)
        if eval(best_hr.split(':')[1].strip()) > best_HR:
            best_batch_size = batch_size
            best_HR = eval(best_hr.split(':')[1].strip())
        sys.stdout.flush()

    print('best parameters: ')
    print({'embedding': str(best_embedding_dim),
           'batch_size': str(best_batch_size),
           })

per_data(data_set='ml-1m')

'''
for data_set in ['ml-1m','pinterest-20']:
    per_data(data_set)
'''

