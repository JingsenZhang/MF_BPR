import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from data import data_utils
import evaluate
from util import utils
from util.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=2, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=8, help="dimension of embedding")
parser.add_argument("--embedding_dim_GMF", type=int, default=8, help="dimension of embedding in GMF submodel")
parser.add_argument("--embedding_dim_MLP", type=int, default=32, help="dimension of embedding in MLP submodel")
parser.add_argument("--hidden_layer_MLP", nargs='*',type=int, default=[32, 16, 8], help="hidden layers in MLP")
parser.add_argument("--use_pretrained", action='store_true', help="use pretrained model to initialize weights")
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


class GMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, dropout):
        super(GMF, self).__init__()
        self.dropout = dropout
        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)
        self.predict_layer = nn.Linear(embedding_dim, 1)
        #self.sigmoid=nn.Sigmoid()


        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        #nn.init.normal_(self.predict_layer.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self,user,item,label):
        prediction=self.predict(user,item)
        criterion=nn.BCEWithLogitsLoss()
        loss=criterion(prediction,label)
        return loss

    def predict(self,user,item):
        embed_user=self.embed_user(user)
        embed_item=self.embed_item(item)
        output=embed_user*embed_item
        prediction=self.predict_layer(output)
        #prediction=self.sigmoid(prediction)
        return prediction.view(-1)

if __name__=="__main__":
    print('GMF')

    #log
    timestamp = time.time()
    run_id = "%.2f" % (timestamp)
    log_dir='result/GMF_log/'+run_id+'.log'
    log_error_dir = 'result/GMF_log/' +run_id+'error.log'
    sys.stdout = Logger(log_dir, sys.stdout)
    sys.stderr = Logger(log_error_dir, sys.stderr)  # redirect std err, if necessary

    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_data, candidate_list, user_num, item_num, train_mat, user_list = data_utils.load_all(data_file)

    train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(candidate_list, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    model = GMF(user_num, item_num, args.embedding_dim, args.dropout)
    model.to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    loss_list = []
    HR, NDCG = [], []
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
            loss=model(user,item,label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/loss', loss.item(), count)
            count += 1
        loss_list.append(loss)

        model.eval()
        HR.clear()
        NDCG.clear()
        hr,ndcg=0,0
        for user, item, label in test_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            prediction = model.predict(user, item)
            _, indices = torch.topk(prediction, args.top_k)
            recommend_u = torch.take(item, indices).cpu().numpy().tolist()

            gt_item = item[0].item()
            HR.append(evaluate.hit(gt_item, recommend_u))
            NDCG.append(evaluate.ndcg(gt_item, recommend_u))
        hr = np.mean(HR)
        ndcg = np.mean(NDCG)
        HR_list.append(hr)
        NDCG_list.append(ndcg)
        print("GMF Epoch: {}, loss: {}, HR: {:.3f}, NDCG: {:.3f}".format(epoch + 1, round(loss.item(), 5), hr, ndcg))

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch+1) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        #print('hr:',hr)
        #print('best_hr:',best_hr)
        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch+1
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'GMF.pth'))

    utils.result_plot(loss_list, 'Training', 'Epochs', 'Loss', "result/GMF_loss.jpg")
    utils.result_plot(HR_list, 'Testing', 'Epochs', 'HR', "result/GMF_HR.jpg")
    utils.result_plot(NDCG_list, 'Testing', 'Epochs', 'NDCG', "result/GMF_NDCG.jpg")
    print("GMF End")
    print('best_hr:{:.3f}'.format(best_hr))
    print('best_ndcg:{:.3f}'.format(best_ndcg))

