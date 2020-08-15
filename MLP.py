import torch
import torch.nn as nn

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
