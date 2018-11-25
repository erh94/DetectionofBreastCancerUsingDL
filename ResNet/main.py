#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import sys

import argparse
import shutil
import time

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

#from torchviz import make_dot
import matplotlib.pyplot as plt
#import graphviz

from pathlib import Path
from userFunctions import *
from userClass import *
from utilsFn import *


# Model Specifics
# import pytorch_resnet as R
import BaseModel as B
import CDDSM
from tqdm import tqdm,tqdm_notebook

import logging
import time
# # TensorBoard Logger

# In[76]:


# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs',comment="pretrained")

experimentName = time.strftime("%d%b%Y%H%M",time.localtime())
suffix = time.strftime("%d%b%Y%H%M",time.localtime())
logtime = time.strftime("%d%b%Y %H:%M:%S",time.localtime())

level = logging.INFO
format = '%(message)s'
handlers = [logging.FileHandler('./logs/Run{}.log'.format(suffix)),logging.StreamHandler()]
logging.basicConfig(level=level,format=format,handlers=handlers)


logging.info("*****************PrerTrained Network Testing******************\n")
# x = torch.randn(batch_size, channels_mammo,heights_mammo , width_mammo)


# # Reading Standard CSV files by TCIA for test/train


#Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device ='cpu'
# Hyper parameters
num_epochs = 100
num_classes = 3
batch_size = 21
learning_rate = 0.001

total_iteration = 10000
img_resize =H=W=512


homedir = str(Path.home())
homedir

train_df = CDDSM.createTrainFrame(homedir)
test_df = CDDSM.createTestFrame(homedir)
mammogram_dir = '/home/himanshu/CuratedDDSM/'
train_file = 'train.csv'
test_file = 'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)

classes = ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')
                


# # Making of CBIS-DDSM Dataset (train,val,test)

# In[66]:


dataset =  CDDSM.MammographyDataset(train_file,homedir,img_resize)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize)

train_dataset , val_dataset = CDDSM.trainValSplit(dataset,val_share=0.1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# # Length of each Dataset

# In[67]:


numberOfTrainData = train_dataset.__len__()
numberOfValData = val_dataset.__len__()
numberOfTestData =  test_dataset.__len__()

total_step=len(train_loader)

logging.info('Size of training dataset {}'.format(numberOfTrainData))
logging.info('Size of Validation dataset {}'.format(numberOfValData))
logging.info('Size of testing dataset {}'.format(numberOfTestData))
logging.info('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))


# # Checking images in each dataset by making grid

# # Get Model

# In[71]:



resnet = models.resnet18(pretrained=True)


# removing last layer of resnet and grad false
features_len = resnet.fc.in_features
# print(features_len)
# pause('Features')
resnet = nn.Sequential(*list(resnet.children())[:-1])


# for param in resnet.parameters():
#     param.requires_grad = False
    
    
class myCustomModel(torch.nn.Module):
    def __init__(self,pretrainedModel,features_len):
        super(myCustomModel,self).__init__()
        
        self.layer0 = nn.Sequential()
        self.layer0.add_module('conv0',nn.Conv2d(1,3,kernel_size=9,stride=1,padding=0,dilation=8))
        self.layer0.add_module('relu0',nn.ReLU())
        self.layer0.add_module('maxpool',nn.MaxPool2d(kernel_size=2))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('pretrained',pretrainedModel)
        self.fc = nn.Linear(in_features=features_len,out_features=3)
    def forward(self,x):
        x = self.layer0(x)
        features = self.layer1(x)
        features = features.view(features.size(0), -1)
        x =  self.fc(features)
        return features , x

def getCustomPretrained(model,features_len):
    return myCustomModel(model,features_len)
    
    
# parameters with parameters requires grad is True
# for p in resnet18.parameters():
#     print(p.requires_grad)

# model = B.getModel(3).to(device)


# In[72]:


# model = B.getModel(3).to(device)
model=getCustomPretrained(resnet,features_len)
model=model.to(device)

# store best prediction in one epoch

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)






# In[77]:




# In[78]:


def save_checkpoint(state,is_best,filename='./models/Unfreezecheckpoint.pth.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'./models/Unfreezemodel_best.pth.tar')


# In[79]:


def train(train_loader,model,criterion,optimizer,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avgaccu =  AverageMeter()
    
    model.train()
    
    end = time.time()
    for i,(images,labels) in enumerate(train_loader):
        data_time.update(time.time()-end)
        images = images.to(device)
        labels = labels.to(device)
        
        #output = model(images)
        _,output = model(images)
        # for resnet returns features and output
        # print(type(output))
        loss = criterion(output,labels)
        
        # top-k ? accuaracy 
        # for now evaluating normal accuracy
        acc = accuracy(output,labels)
        
        #loss.item() to get the loss value from loss tensor
        losses.update(loss.item(), images.size(0))
        avgaccu.update(acc,images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        logging.info('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, acc=avgaccu))


# In[80]:


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avgaccu =  AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input=input.to(device)
            target =  target.to(device)
            
            # compute output
            _,output = model(input)
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc= accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            avgaccu.update(acc,input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logging.info('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,acc=avgaccu))

        logging.info(' * Acc {acc.avg:.3f}'
              .format(acc=avgaccu))

    return acc


# In[81]:


def accuracy(output,target):
    with torch.no_grad():
        batch_size =  target.size(0)
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        acc = correct/total
    return acc


# In[82]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[83]:


def test(test_loader,model):

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            _,outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logging.info('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# In[84]:


def adjust_learning_rate(optimizer,epoch,initLR):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epoch'''
    lr = initLR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# # Training

# In[86]:


best_acc = 0

log_freq=10
for epoch in range(num_epochs):
    
    adjust_learning_rate(optimizer,epoch,learning_rate)
    
    
    train(train_loader,model,criterion,optimizer,epoch+1)
    
    
    
    acc =  validate(val_loader,model,criterion)
    
    
    
    is_best = acc > best_acc
    
    best_acc = max(acc,best_acc)
    
    
    #saving the checkpoint if is_best is True
    save_checkpoint({
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_acc':best_acc,
        'optimizer':optimizer.state_dict(),
    },is_best)
    
    if(epoch % log_freq==0):
        test(test_loader=test_loader,model=model)
        
test(test_loader,model)


# In[ ]:





# In[ ]:




