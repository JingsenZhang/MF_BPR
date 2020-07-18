import os
import torch
import numpy as np
#import matplotlib as mpl      #服务器上使用时
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl


def result_plot(result,title,xlable,ylable,save_path='image/result.jpg'):
    plt.plot(result)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.show()
    #plt.savefig(save_path)
    #plt.clf()  # 画完第一个图之后重置

#模型保存
def save_model(model, model_name, model_path=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    if model_path is None:
        model_path = "models/model_{}".format(model_name)
    print('Saving models to {}'.format(model_path))
    torch.save(model.state_dict(), model_path)

#模型加载
def load_model(model,model_path):
    print('Loading models from {}'.format(model_path))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

#数据写入txt(方便画图)
def save_txt(save_path,result):
    np.savetxt(save_path, result)
    #数据加载：np.loadtxt(path)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
