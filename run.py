import os

if __name__=='__main__':
    os.system( 'python main.py --model BPR --lr 0.1 --lambd 1e-5 --batch_size 4096 --num_epoch 2 --top_k 10 --factor_dim 20 --num_ng 3 --test_samples_num 100')











    #os.system('python MF_run.py --factor_dim 20 --lambd 1e-5 --lr 0.1 --num_epoch 2')   #参数类型在parser中定义
    #os.system('python BPR_run.py --lr 0.01 --lambd 0.001 --batch_size 4096 --num_epoch 2 --factor_dim 30 --num_ng 4 --test_num_ng 99')
    #另外两种格式化方法
    #os.system("python MF_run.py --factor_num %d --lambd %f --lr %f --num_epoch %d" % (20,1e-5,0.1,1))
    #os.system("python MF_run.py --factor_num {} --lambd {} --lr {} --num_epoch {}".format(20,1e-5,0.1,1))