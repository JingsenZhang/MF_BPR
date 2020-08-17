import os
import time
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter

from data import data_utils
import evaluate
from util import utils
from util.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim_GMF", type=int, default=8, help="dimension of embedding in GMF submodel")
parser.add_argument("--embedding_dim_MLP", type=int, default=32, help="dimension of embedding in MLP submodel")
parser.add_argument("--hidden_layer_MLP", nargs='*',type=int, default=[32, 16, 8], help="hidden layers in MLP")
parser.add_argument("--pretrained", action='store_true', help="use pretrained model to initialize weights")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="./dataset/")
parser.add_argument("--model_path", type=str, default="./result/")
parser.add_argument("--out", type=bool, default=True, help="save model or not")
parser.add_argument("--disable_cuda",action='store_true', help="Disable CUDA")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
cudnn.benchmark = True


class NeuMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim_GMF, embedding_dim_MLP, hidden_layer_MLP,
                 dropout, GMF_model=None, MLP_model=None):
        super(NeuMF, self).__init__()
        self.dropout = dropout
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, embedding_dim_GMF)
        self.embed_item_GMF = nn.Embedding(item_num, embedding_dim_GMF)
        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP)


        MLP_modules = []
        self.num_layers = len(hidden_layer_MLP)
        print('hidden:', hidden_layer_MLP)
        print('hidden[0]:', hidden_layer_MLP[0])
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MLP_modules.append(nn.Linear(embedding_dim_MLP*2, hidden_layer_MLP[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer_MLP[i-1], hidden_layer_MLP[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer_MLP[-1] + embedding_dim_GMF, 1)

        print('pretrained:',args.pretrained)
        if not args.pretrained:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            # Kaiming/Xavier initialization can not deal with non-zero bias terms
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self,user,item,label):
        prediction=self.predict(user,item)
        criterion=nn.BCEWithLogitsLoss()
        loss=criterion(prediction,label)
        return loss

    def predict(self,user,item):
        embed_user_GMF=self.embed_user_GMF(user)
        embed_item_GMF=self.embed_item_GMF(item)
        output_GMF=embed_user_GMF*embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        iteraction=torch.cat((embed_user_MLP,embed_item_MLP),-1)
        output_MLP=self.MLP_layers(iteraction)

        output=torch.cat((output_GMF,output_MLP),-1)
        prediction=self.predict_layer(output)
        return prediction.view(-1)

if __name__=="__main__":
    print('NeuMF')

    #log
    timestamp = time.time()
    run_id = "%.2f" % (timestamp)
    log_dir='result/NeuMF_log/'+run_id+'.log'
    log_error_dir = 'result/NeuMF_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary

    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_data, candidate_list, user_num, item_num, train_mat,user_list = data_utils.load_all(data_file)

    train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(candidate_list, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    GMF_model_path = os.path.join(args.model_path, 'GMF.pth')
    MLP_model_path = os.path.join(args.model_path, 'MLP.pth')
    if args.pretrained:
        assert os.path.exists(GMF_model_path), 'lack of GMF model'
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    print('arg.hidden:',args.hidden_layer_MLP)
    model = NeuMF(user_num, item_num, args.embedding_dim_GMF, args.embedding_dim_MLP,
                  args.hidden_layer_MLP, args.dropout, GMF_model, MLP_model)
    model.to(device=args.device)
    if args.pretrained:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    loss_list = []
    HR_list = []
    NDCG_list = []

    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.float().to(device=args.device)

            model.zero_grad()
            loss = model(user, item, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1
        loss_list.append(loss)

        model.eval()
        HR,NDCG=[],[]
        for user,item,label in test_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            prediction=model.predict(user,item)
            _,indices=torch.topk(prediction,args.top_k)
            recommend_u=torch.take(item,indices).cpu().numpy().tolist()
            gt_item=item[0].item()
            HR.append(evaluate.hit(gt_item,recommend_u))
            NDCG.append(evaluate.ndcg(gt_item,recommend_u))
        hr=np.mean(HR)
        ndcg=np.mean(NDCG)
        HR_list.append(hr)
        NDCG_list.append(ndcg)
        print("NeuMF Epoch: {}, loss: {}, HR: {}, NDCG: {}".format(epoch + 1, round(loss.item(), 5), round(hr, 5),round(ndcg,5)))

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch+1) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch+1
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, os.path.join(args.model_path, 'MLP.pth'))

    #print("NeuMF End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
    utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/NeuMF_loss.jpg")
    utils.result_plot(HR_list, 'Testing', 'Epochs', 'HR', "result/NeuMF_HR.jpg")
    utils.result_plot(NDCG_list, 'Testing', 'Epochs', 'NDCG', "result/NeuMF_NDCG.jpg")
    print("NeuMF End")
    print('best_hr:{:.3f}'.format(best_hr))
    print('best_ndcg:{:.3f}'.format(best_ndcg))