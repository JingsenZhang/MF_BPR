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
        if 'args:' in line:
            parameter = line
        if 'final best performance:' in line:
            result = line
        if 'Experiment cost:' in line:
            time_cost = line
    return parameter, result, time_cost


def per_data(data_path):
    print(' embedding_dim ')
    best_f1 = 0.0
    for embedding_dim in [10, 20, 30, 40, 50]:
        cmd = 'python3 BPR.py ' + \
              ' --data_path ' + data_path + ' ' + \
              ' --embedding_dim ' + str(embedding_dim) + ' ' + \
              ' --batch_size 1024 '
        print(cmd)
        parameter, result, time_cost = sh(cmd)
        print(str(parameter))
        print(time_cost)
        print(result)
        if eval(result.split(':')[1].strip())[2] > best_f1:
            best_embedding_dim = embedding_dim
            best_f1 = eval(result.split(':')[1].strip())[2]
        sys.stdout.flush()


    print(' batch_size ')
    best_f1 = 0.0
    for batch_size in [64, 128, 256, 512, 1024]:
        cmd = 'python3 BPR.py ' + \
              ' --data_path ' + data_path + ' ' + \
              ' --embedding_dim ' + str(best_embedding_dim) + ' ' + \
              ' --batch_size ' + str(batch_size)
        print(cmd)
        parameter, result, time_cost = sh(cmd)
        print(str(parameter))
        print(time_cost)
        print(result)
        if eval(result.split(':')[1].strip())[2] > best_f1:
            best_batch_size = batch_size
            best_f1 = eval(result.split(':')[1].strip())[2]
        sys.stdout.flush()


    print('best parameters: ')
    print({'embedding': str(best_embedding_dim),
           'batch_size': str(best_batch_size),
           })



for data in ['office', 'Instant', 'Digital', 'baby', 'pet', 'phone']:
    data_path = "./data/amazon/" + data + "/"
    print(data_path)
    per_data(data_path)

