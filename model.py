import torch
import torch.nn as nn


class MF(nn.Module):    #BiasMF
    def __init__(self, user_num, item_num, factor_dim):
        super(MF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_dim = factor_dim

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        self.user_bias = nn.Embedding(user_num, 1)
        self.user_bias.weight.data = torch.zeros(self.user_num, 1).float()   #初始化偏置项
        self.item_bias = nn.Embedding(item_num, 1)
        self.item_bias.weight.data = torch.zeros(self.item_num, 1).float()

    def forward(self,user_indices, item_indeices, global_mean,rates):
        rates_y = self.predict(user_indices, item_indeices, global_mean)
        criterion = nn.MSELoss()
        loss = criterion(rates_y, rates)
        return loss

    def predict(self,user_indices, item_indeices, global_mean):
        user_vec = self.user_emb(user_indices)           #pu向量
        item_vec = self.item_emb(item_indeices)          #qi向量

        dot = torch.mul(user_vec, item_vec).sum(dim=1)   #在1维度上求和（即遍历隐类k）

        rates = dot + self.user_bias(user_indices).view(-1) + self.item_bias(item_indeices).view(-1) + global_mean    #BiasMF
        return rates

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_dim):
        super(BPR, self).__init__()

        self.user_emb = nn.Embedding(user_num, factor_dim)
        self.item_emb = nn.Embedding(item_num, factor_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self,user,item_i,item_j):
        prediction_i = self.predict(user,item_i)
        prediction_j = self.predict(user,item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        return loss

    def predict(self,user,item):                #参数可为数值或list
        user = self.user_emb(user)
        item = self.item_emb(item)
        prediction = (user * item).sum(dim=-1)
        return prediction


class GMF(nn.Module):
    def __init__(self,user_num,item_num,factor_dim,dropout):
        super(GMF,self).__init__()
        self.dropout=dropout
        self.embed_user=nn.Embedding(user_num,factor_dim)
        self.embed_item=nn.Embedding(item_num,factor_dim)
        self.predict_layer=nn.Linear(factor_dim,1)

        nn.init.normal_(self.embed_user.weight,std=0.01)
        nn.init.normal_(self.embed_item.weight,std=0.01)
        nn.init.kaiming_normal_(self.predict_layer.weight,a=1,nonlinearity='sigmoid')   #???

        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self,user,item,label):
        prediction=self.predict(user,item)
        criterion=nn.BCEWithLogitsLoss()
        loss=criterion(prediction,label)
        return loss

    def predict(self,user,item):
        user=self.embed_user(user)
        item=self.embed_item(item)
        output=user*item
        prediction=self.predict_layer(output)
        return prediction.view(-1)

class MLP(nn.Module):
    def __init__(self,user_num,item_num,factor_dim,hidden_layer,dropout):
        super(MLP,self).__init__()
        self.dropout=dropout
        self.embed_user=nn.Embedding(user_num,factor_dim)
        self.embed_item=nn.Embedding(item_num,factor_dim)

        MlP_modules=[]
        self.num_layers=len(hidden_layer)
        for i in range(self.num_layers):
            MlP_modules.append(nn.Dropout(p=self.dropout))
            if i==0:
                MlP_modules.append(nn.Linear(factor_dim*2,hidden_layer[0]))
            else:
                MlP_modules.append(nn.Linear(hidden_layer[i-1],hidden_layer[i]))
            MlP_modules.append(nn.ReLU())
        self.MLP_layers=nn.Sequential(*MlP_modules)

        self.predict_layer=nn.Linear(hidden_layer[-1],1)

        nn.init.normal_(self.embed_user.weight,std=0.01)
        nn.init.normal_(self.embed_item.weight,std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self,user,item,label):
        prediction=self.predict(user,item)
        criterion=nn.BCEWithLogitsLoss()
        loss=criterion(prediction,label)
        return loss

    def predict(self,user,item):
        user=self.embed_user(user)
        item=self.embed_item(item)
        #print('user_shape:',user.size())
        #print('item_shape:',item.size())
        interaction=torch.cat((user,item),-1)
        output=self.MPL_layer(interaction)
        prediction=self.predict_layer(output)
        return prediction.view(-1)

class NeuMF(nn.Module):
    def __init__(self,user_num,item_num,factor_dim_GMF,factor_dim_MLP,hidden_layer_MLP,dropout,pre_training,GMF_model=None,MLP_model=None):
        super(NeuMF,self).__init__()
        self.dropout=dropout
        self.GMF_model=GMF_model
        self.MLP_model=MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_dim_GMF)
        self.embed_item_GMF = nn.Embedding(item_num, factor_dim_GMF)
        self.embed_user_MLP = nn.Embedding(user_num, factor_dim_MLP)
        self.embed_item_MLP = nn.Embedding(item_num, factor_dim_MLP)

        MlP_modules = []
        self.num_layers = len(hidden_layer_MLP)
        for i in range(self.num_layers):
            MlP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MlP_modules.append(nn.Linear(factor_dim_MLP * 2, hidden_layer_MLP[0]))
            else:
                MlP_modules.append(nn.Linear(hidden_layer_MLP[i - 1], hidden_layer_MLP[i]))
            MlP_modules.append(nn.ReLU())
        self.MPL_layers = nn.Sequential(*MlP_modules)

        self.predict_layer=nn.Linear(hidden_layer_MLP[-1]+factor_dim_GMF,1)

        if not pre_training:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            for m in self.MPL_layers:
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,a=1,nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m,nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item.weight)
            for (m1,m2) in zip(self.MPL_layers,self.MLP_model.MLP_layers):
                if isinstance(m1,nn.Linear) and isinstance(m2,nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            predict_weight=torch.cat((self.GMF_model.predict_layer.weight,self.MLP_model.predict_layer.weight),1)
            predict_bias=self.GMF_model.predict_layer.bias+self.MLP_model.predict_layer.bias
            self.predict_layer.weight.data.copy_(0.5*predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)

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
        output_MLP=self.MPL_layers(iteraction)

        output=torch.cat((output_GMF,output_MLP),-1)
        prediction=self.predict_layer(output)
        return prediction.view(-1)
