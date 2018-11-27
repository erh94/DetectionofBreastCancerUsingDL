#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm
from utilsFn import *

import BaseModel as B
import CDDSM
import time
import logging
import copy


comment = 'Proper training, no resumes used , resnet18 with pretrained weights on imagenet is used, transfer learning to classify num_classes'

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args =  parser.parse_args()
# # Hyper Parameters







# In[ ]:


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Selected Device {}'.format(device))


# Hyper parameters
num_epochs = 100
num_classes = 3
batch_size = 10
learning_rate = 0.005

total_iteration = 10000
img_resize =H=W=512

homedir = str(Path.home())

experimentName = time.strftime("%d%b%Y%H%M",time.localtime())


# # CSV preprocessing

# In[ ]:


train_df,train_classes = CDDSM.createTrainFrame(homedir)
test_df,test_classes = CDDSM.createTestFrame(homedir)

assert (train_classes==test_classes), 'Classes are different'
num_classes = train_classes
print('Number of classes : {}'.format(num_classes))
mammogram_dir = '/home/himanshu/CuratedDDSM/'
train_file = 'train.csv'
test_file = 'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)




# # Dataset

# In[ ]:

dataset =  CDDSM.MammographyDataset(train_file,homedir,img_resize)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize)

train_dataset , val_dataset = CDDSM.trainValSplit(dataset,val_share=0.1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=2*batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1, 
                                          shuffle=False)


# # Length of each Dataset

# In[67]:


numberOfTrainData = train_dataset.__len__()
numberOfValData = val_dataset.__len__()
numberOfTestData =  test_dataset.__len__()

total_step=len(train_loader)


# In[ ]:



prefix = 'E{}B{}Im{}lr{}'.format(num_epochs,batch_size,H,learning_rate)
print(prefix)
trainLogFile = openfile('./logs/'+experimentName+'/train{}.csv'.format(prefix))
valLogFile = openfile('./logs/'+experimentName+'/validation{}.csv'.format(prefix))
testLogFile = openfile('./logs/'+experimentName+'/test{}.csv'.format(prefix))
infoLogFile = openfile('./logs/'+experimentName+'/info{}.txt'.format(prefix))
setup_logger('trainlog',trainLogFile)
setup_logger('testlog',testLogFile)
setup_logger('vadLog',valLogFile)
setup_logger('infoFile',infoLogFile)

logger('Step|length,BatchTime,BatchAvgTime,Loss,AvgLoss,Accuracy,AverageAccuracy','info','vadLog')
logger('Epoch(step|length),BatchTime,AvgBatchTime,Loss,AvgLoss,Accuracy,AvgAccuracy','info','trainlog')
logger('index,predicted,true,path,Accuracy','info','testlog')
logger('Size of training dataset {}'.format(numberOfTrainData),'info','infoFile')
logger('Size of Validation dataset {}'.format(numberOfValData),'info','infoFile')
logger('Size of testing dataset {}'.format(numberOfTestData),'info','infoFile')
logger('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step),'info','infoFile')


######### Additinal info Logs ########
logger('number of classes :{}'.format(num_classes),'info','infoFile')
logger('comment :{}'.format(comment),'info','infoFile')


###################### SAVE CHECKPOINT ############################



def save_checkpoint(state,is_best,filename=openfile('./models/{}/checkpoint{}.pth.tar'.format(experimentName,prefix))):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,openfile('./models/{}/best_{}.pth.tar'.format(experimentName,prefix)))



################ Pretrained Resnet Model ####################
resnet = models.resnet18(pretrained=True)
features_len = resnet.fc.in_features
resnet = nn.Sequential(*list(resnet.children())[:-1])



# for param in resnet.parameters():
#     param.requires_grad = False
    
    
class myCustomModel(torch.nn.Module):
    def __init__(self,pretrainedModel,features_len,classes):
        super(myCustomModel,self).__init__()
        
        self.layer0 = nn.Sequential()
        self.layer0.add_module('conv0',nn.Conv2d(1,3,kernel_size=9,stride=1,padding=0,dilation=8))
        self.layer0.add_module('relu0',nn.ReLU())
        self.layer0.add_module('maxpool',nn.MaxPool2d(kernel_size=2))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('pretrained',pretrainedModel)
        self.fc = nn.Linear(in_features=features_len,out_features=classes)
    def forward(self,x):
        x = self.layer0(x)
        features = self.layer1(x)
        features = features.view(features.size(0), -1)
        x =  self.fc(features)
        return features , x

def getCustomPretrained(model,features_len,classes):
    return myCustomModel(model,features_len,classes)
    
    
# parameters with parameters requires grad is True
# for p in resnet18.parameters():
#     print(p.requires_grad)

# model = B.getModel(3).to(device)


# In[72]:


# model = B.getModel(3).to(device)
model=getCustomPretrained(resnet,features_len,num_classes)
model=model.to(device)

# store best prediction in one epoch

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


model.layer0.apply(init_weights)
##################### Model End , Optimizer, Loss End ##########################


#################### start epoch and best_acc ##############################

start_epoch = 0
best_acc = 0



if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        logger('Resumed from {}'.format(args.resume),'info','infoFile')
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        start_epoch=0

######################## RESUME END #################################

########################## TRAINING ##################################

def train(train_loader,model,criterion,optimizer,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avgaccu =  AverageMeter()
    
    model.train()
    
    end = time.time()
    for i,(images,labels,_) in enumerate(train_loader):
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

        logger('[{0}][{1}/{2}],'
              '{batch_time.val:.3f},{batch_time.avg:.3f},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{acc.val:.4f},{acc.avg:.4f}'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, acc=avgaccu),'info','trainlog')

################################### Validation #######################################

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avgaccu =  AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target,_) in enumerate(val_loader):
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
            print('Validation')
            logger('[{0}/{1}],'
                  '{batch_time.val:.3f},{batch_time.avg:.3f},'
                  '{loss.val:.4f},{loss.avg:.4f}),'
                  '{acc1.val:.3f},{acc1.avg:.3f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,acc1=avgaccu),'info','vadLog')

        print(' **********Validation Accuracy {acc1.avg:.3f}*****************'
              .format(acc1=avgaccu))

    return avgaccu.avg



############################ TEST #######################################################



def test(model,test_loader,epoch):
    model_test = copy.deepcopy(model)
    #model_test
    model_test.eval()
    
    avgacc = AverageMeter()
    
    with torch.no_grad():
        for i,(images,labels,path) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            _,outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            
            acc = accuracy(outputs,labels)
            avgacc.update(acc)
            

            logger('{}/{},{},{},{},{}'.format(i,epoch,predicted.cpu().numpy()[0],
                                              labels.cpu().numpy()[0],str(path),avgacc.avg),
                   'info','testlog')



########################## TRAINING LOOP ###############################################

for epoch in range(start_epoch , num_epochs):
       
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
    
    if(is_best):
        test(model,test_loader,epoch+1)




