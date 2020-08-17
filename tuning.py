import subprocess
import numpy as np
import sys

def run(command):
    subprocess.call(command, shell=True)

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, time_cost = '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'best_hr:' in line:
            best_hr = line
        if 'best_ndcg:' in line:
            best_ndcg = line
    return  best_hr, best_ndcg


def per_data():
    parameter='embedding_dim'
    best_HR,best_NDCG=0,0
    for embedding_dim in [10, 20, 30, 40, 50]:
        cmd = 'python GMF.py ' + \
              ' --embedding_dim ' + str(embedding_dim) + ' ' + \
              ' --batch_size 256 '
        print(cmd)
        best_hr,best_ndcg = sh(cmd)
        print(str(parameter))
        print(best_hr)
        print(best_ndcg)
        if eval(best_hr.split(':')[1].strip()) > best_HR:
            best_embedding_dim = embedding_dim
            best_HR = eval(best_hr.split(':')[1].strip())
        sys.stdout.flush()


    print(' batch_size ')
    best_HR, best_NDCG = 0, 0
    for batch_size in [64, 128, 256, 512, 1024]:
        cmd = 'python GMF.py ' + \
              ' --embedding_dim ' + str(best_embedding_dim) + ' ' + \
              ' --batch_size ' + str(batch_size)
        print(cmd)
        best_hr,best_ndcg = sh(cmd)
        print(str(parameter))
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

per_data()








'''
for data in ['office', 'Instant', 'Digital', 'baby', 'pet', 'phone']:
    data_path = "./data/amazon/" + data + "/"
    print(data_path)
    per_data(data_path)
'''

