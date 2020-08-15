import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self,user_num,item_num,factor_dim_GMF,factor_dim_MLP,hidden_layer_MLP,dropout,pre_training,GMF_model=None,MLP_model=None):
        super(NCF,self).__init__()
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



