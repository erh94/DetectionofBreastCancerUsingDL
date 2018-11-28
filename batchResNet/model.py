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
        self.layer0 = nn.Sequential()
        self.layer0.add_module('conv0',nn.Conv2d(1,3,kernel_size=9,stride=1,padding=0,dilation=8))
        self.layer0.add_module('relu0',nn.ReLU())
        self.layer0.add_module('maxpool',nn.MaxPool2d(kernel_size=2))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('pretrained',pretrainedModel)
        self.fc = nn.Linear(in_features=512,out_features=num_classes)
    def forward(self,x):
        x = self.layer0(x)
        features = self.layer1(x)
        features = features.view(features.size(0), -1)
        x =  self.fc(features)
        return x, self.prediction(x)
    def prediction(self,x):
        predBag=1-torch.prod(torch.exp(torch.mul(F.relu(x),-0.001)),dim=0)
        return predBag

def getCustomPretrained(model,num_classes):
    return myCustomModel(model,num_classes)
    

class MIL_loss(torch.nn.Module):
    ''' MIL Loss Layer'''
    def __init__(self,lamda):
        super(MIL_loss, self).__init__()
        self.lamda = lamda
        
    def forward(self,scores,labels):
        ''' lamda is postive constant,
        to convert scores(h_i) into probability'''
        sum_scores = -torch.sum(torch.mul(F.relu(scores),self.lamda),dim=0)
        loss = -torch.dot(1-labels.float(),sum_scores.float())
        return loss
