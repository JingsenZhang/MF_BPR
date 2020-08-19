import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



if __name__=='__main__':
    #os.system('python MF.py --lr 0.01 --epochs 2 --embedding_dim 32 --lambd 1e-5')
    #os.system( 'python BPR.py --lr 0.01  --batch_size 256 --epochs 2 --top_k 10 --embedding_dim 8 --num_ng 4 --lambd 1e-5')
    #os.system( 'python GMF.py --lr 0.001 --batch_size=256 --epochs 2 --top_k 10 --embedding_dim 8 --num_ng 4 --data_set ml-1m --out True ')
    #os.system('python MLP.py --lr 0.001 --batch_size=256 --epochs 2 --top_k 10 --embedding_dim 32 --num_ng 4 --data_set ml-1m --out True ')
    os.system('python NeuMF.py --lr 0.001 --batch_size=256 --epochs 2 --top_k 10 --pretrained --embedding_dim_GMF 8 --embedding_dim_MLP 32 --hidden_layer_MLP 32 16 8 --num_ng 4 --data_set ml-1m --out True ')
    #使用预训练的neumf,记得更改图片名
















    #os.system('python MF_run.py --factor_dim 20 --lambd 1e-5 --lr 0.1 --num_epoch 2')   #参数类型在parser中定义
    #os.system('python BPR_run.py --lr 0.01 --lambd 0.001 --batch_size 4096 --num_epoch 2 --factor_dim 30 --num_ng 4 --test_num_ng 99')
    #另外两种格式化方法
    #os.system("python MF_run.py --factor_num %d --lambd %f --lr %f --num_epoch %d" % (20,1e-5,0.1,1))
    #os.system("python MF_run.py --factor_num {} --lambd {} --lr {} --num_epoch {}".format(20,1e-5,0.1,1))