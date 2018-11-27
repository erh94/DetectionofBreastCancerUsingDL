#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import math
import torch
import torchvision
import argparse
from pathlib import Path

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


parser = argparse.ArgumentParser()

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args =  parser.parse_args()


# # Hyper Parameters

# In[ ]:


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Device {}'.format(device))


num_epochs = 50
num_classes = 3
batch_size = 10
learning_rate = 0.005


#Image size
img_resize=H=W=512

homedir = str(Path.home())

experimentName = time.strftime("%d%b%Y%H%M",time.localtime())


# # CSV preprocessing

# In[ ]:


train_df = CDDSM.createTrainFrame(homedir)
test_df = CDDSM.createTestFrame(homedir)
mammogram_dir = '/home/himanshu/CuratedDDSM/'
train_file = 'train.csv'
test_file = 'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)

classes = ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')


# # Dataset

# In[ ]:


train_dataset =  CDDSM.MammographyDataset(train_file,homedir,img_resize)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1, 
                                          shuffle=False)

number_of_training_data = train_dataset.__len__()
number_of_testing_data = test_dataset.__len__()


total_step = len(train_loader)







print('Size of training dataset {}'.format(number_of_training_data))
print('Size of testing dataset {}'.format(number_of_testing_data))
print('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))


# In[ ]:


model = B.getModel(3).to(device)
# getModel gives a model for images 512*512
# getModel1024 gives model for images 1024*1024
# getModel1024L gives model for images 1024*1024

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



<<<<<<< HEAD

############################# RESUmE ################################
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        #start_epoch = checkpoint['epoch']
        #best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch 0)".format(args.resume))
        logger('Resumed from {}'.format(args.resume),'info','infoFile')
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        start_epoch=0

######################## RESUME END #################################

=======
>>>>>>> ee7f400288581bd9084b183d4a3b345a1e0d81ac
def init_weights(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
model.apply(init_weights)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def test(model,test_loader,epoch):
    model_test = copy.deepcopy(model)
    model_test.eval()
    
    avgacc = AverageMeter()
    
    with torch.no_grad():
        for i,(images,labels,path) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            
            acc = accuracy(outputs,labels)
            avgacc.update(acc)
            

            logger('{}/{},{},{},{},{}'.format(i,epoch,predicted.cpu().numpy()[0],
                                              labels.cpu().numpy()[0],str(path),avgacc.avg),
                   'info','testlog')



# In[ ]:





# # Logger

# In[ ]:


prefix = 'E{}B{}Im{}lr{}'.format(num_epochs,batch_size,H,learning_rate)
print(prefix)
trainLogFile = openfile('./logs/'+experimentName+'/train{}.csv'.format(prefix))
# valLogFile = openfile('./logs/'+experimentName+'/validation{}.csv'.format(prefix))
testLogFile = openfile('./logs/'+experimentName+'/test{}.csv'.format(prefix))
setup_logger('trainlog',trainLogFile)
setup_logger('testlog',testLogFile)


logger('Epoch,Step,Loss,Accuracy','info','trainlog')
logger('index,predicted,true,path,Accuracy','info','testlog')


# In[ ]:


def accuracy(outputs,labels):
    total=0
    correct=0
    _,predicted = torch.max(outputs.data,1)
    total+=labels.size(0)
    correct += (predicted==labels).sum().item()
    acc = correct/total
    return acc


# In[ ]:


def adjust_learning_rate(optimizer,epoch,initLR):
    '''Sets the learning rate to the initial LR decayed by 10 every 50 epoch'''
    lr = initLR * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[ ]:


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


# In[ ]:


best_acc = 0
for epoch in range(num_epochs):
    
    avgacc = AverageMeter()
    
    adjust_learning_rate(optimizer,epoch+1,learning_rate)
    
    for i,(images,labels,_) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        acc = accuracy(outputs,labels)
        
        avgacc.update(acc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger('[{}/{}],[{}/{}],{:.4f},{:.4f}'.format(epoch+1,
                                              num_epochs,i+1,
                                              total_step,
                                              loss.item(),
                                              avgacc.avg),'info','trainlog')
    is_best = avgacc.avg > best_acc
    
    best_acc = max(acc,best_acc)
    
    if(is_best):
        print("Accuracy :{}".format(avgacc.avg))
        torch.save(model.state_dict(),openfile('./models/{}/model_{}.ckpt'.format(experimentName,
                                                                    prefix)))
        test(model,test_loader,epoch+1)


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




