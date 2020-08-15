import torch
import torch.nn as nn

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

