import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


class myCustomModel(torch.nn.Module):
    def __init__(self,pretrainedModel,num_classes):
        super(myCustomModel,self).__init__()
        self.num_classes=num_classes
        # self.layer0 = nn.Sequential()
        # self.layer0.add_module('conv0',nn.Conv2d(1,3,kernel_size=9,stride=1,padding=0,dilation=8))
        # self.layer0.add_module('relu0',nn.ELU())
        # self.layer0.add_module('maxpool',nn.MaxPool2d(kernel_size=2))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('pretrained',pretrainedModel)
        self.fc = nn.Linear(in_features=512,out_features=num_classes)
        self.Sf = nn.Softmax(dim=1)
    def forward(self,x):
        
        features = self.layer1(x)
        features = features.view(features.size(0), -1)
        x =  self.Sf(self.fc(features))
        return x, self.prediction(x)
    def prediction(self,x):
        predBag=1-torch.prod(1-x,dim=0)
        return predBag

def getCustomPretrained(model,num_classes):
    return myCustomModel(model,num_classes)
    

class MIL_loss(torch.nn.Module):
    ''' MIL Loss Layer'''
    def __init__(self,lamda):
        super(MIL_loss, self).__init__()
        self.lamda = lamda
        
    def forward(self,input,labels):
        ''' lamda is postive constant,
        to convert scores(h_i) into probability'''
        #print(input)
        predBag,_=torch.max(torch.add(input,-1),dim=0)
        predBag = F.softmax(torch.add(predBag,-1),dim=0)        

        #assert False, 'prinst'
        #sum_scores = -torch.sum(torch.mul(F.elu(scores),self.lamda),dim=0)
        #print(torch.max(F.elu(scores.detach()),dim=0),scores.shape)
        #a,b = torch.max(scores,dim=0)
        #print(a,a.shape)
        #sum_scores = - torch.mul((torch.max(F.elu(scores),dim=0)),1)
        #print(sum_scores,sum_scores.shape)
        loss=0
        for yhat,y in zip(predBag,labels):
            if y==1:
                loss+= -torch.log(yhat)
            else:
                loss+= -torch.log(1-yhat)
        #loss = -torch.dot(1-labels.float(),sum_scores.float())
        #print(labels)      


        
        return loss
