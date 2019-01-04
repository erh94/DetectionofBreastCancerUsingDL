#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from pathlib import Path
from customDataset import *
from tqdm import tqdm
from utilsFn import *
from model import *


# # Hyper Parameters

# In[2]:
experimentName = time.strftime("%d%b%Y%H%M",time.localtime())
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/{}'.format(experimentName))

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args =  parser.parse_args()



numEpochs = 800
batchSize = 1
lr = 0.005
classes = 2
patchSize = H = W =224
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

homedir = str(Path.home())
trainFile = 'train.csv'
testFile = 'test.csv'
experimentName = time.strftime("%d%b%Y%H%M",time.localtime())


# # Dataset

# In[3]:


dataset = MammographyDataset(trainFile,homedir,patchSize,mode='train')
testDataset = MammographyDataset(testFile,homedir,patchSize,mode='test')
trainDataset , valDataset = trainValSplit(dataset,val_share=0.20)
trainLoader = DataLoader(trainDataset,batchSize,shuffle=True)
valLoader = DataLoader(valDataset,batchSize,shuffle=True)
#------------- test transformation should be stop-------------#
testLoader = DataLoader(testDataset,batchSize,shuffle=False)

numberOfTrainData = trainDataset.__len__()
numberOfValData = valDataset.__len__()
numberOfTestData =  testDataset.__len__()

total_step=len(trainLoader)


# In[4]:


resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
model = getCustomPretrained(resnet,classes)

criterion = MIL_loss(0.001)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)

model = model.to(device)







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




def accuracyperImage(pred,labels):
    with torch.no_grad():
        pred_class =  torch.zeros_like(pred)
        pred_class[pred.argmax()]=1
        if(torch.equal(pred_class,labels)):
            return 1
        else:
            return 0


# In[10]:


def basiclogging(str):
    logger('*******************MIL CNN*********************','info',str)
    logger('Size of training dataset {}'.format(numberOfTrainData),'info',str)
    logger('Size of Validation dataset {}'.format(numberOfValData),'info',str)
    logger('Size of testing dataset {}'.format(numberOfTestData),'info',str)
    
    if(str=='trainlog'):
        logger('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Patch size {}*{}\n Step {}'
            .format(numEpochs,batchSize,lr,H,W,total_step),'info',str)
        logger('Epoch\tBatchTime\tAverageBatchTime\tLoss\tAverageLoss\tAccuracy\tAverageAccuracy','info',str)
    if(str=='vadLog'):
        logger('Epoch\tBatchTime\tAverageBatchTime\tLoss\tAverageLoss\tAccuracy\tAverageAccuracy','info',str)
    if(str=='testlog'):
        logger('Epoch\tAccuracy\tAverageAccuracy\tTruePositive\tTrueNegative\tFalsePositive\tFalseNegative\tPath','info',str)



# In[12]:


def adjust_learning_rate(optimizer,epoch,initLR):
    '''Sets the learning rate to the initial LR decayed by 10 every 50 epoch'''
    lr = initLR * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[13]:


def save_checkpoint(state,is_best,filename=openfile('./models/'+experimentName+'/checkpoint.pth.tar')):
        torch.save(state,filename)
        if is_best:
            shutil.copyfile(filename,openfile('./models/'+experimentName+'/model_best.pth.tar'))


# In[14]:


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




def confusionMatrix(pred,labels):
    tp=0
    tn=0
    fp=0
    fn=0
    
    a=labels.argmax().cpu().numpy()
    b=pred.argmax().cpu().numpy()

    if(a==0 and b==0):
        tn=1
    elif(a==0 and b==1):
        fp=1
    elif(a==1 and b==0):
        fn=1
    else:
        tp=1
        
    return (tp,fp,fn,tn)
    


# In[21]:


def test(testLoader,model,epoch):
    truepositive = AverageMeter()
    truenegative = AverageMeter()
    falsepositive = AverageMeter()
    falsenegative = AverageMeter()
    accuracy = AverageMeter()
    
    true = []
    predicted = []
    model.eval()
    with torch.no_grad():
        for i,(batch,bag_label,path) in enumerate(testLoader):
            images = batch[0].to(device)
            labels = bag_label[0].to(device)

            _,pred = model(images)
            
            acc = accuracyperImage(pred,labels)
            
            tp,fp,fn,tn = confusionMatrix(pred,labels)
            a=labels.argmax().cpu().numpy()
            b=pred.argmax().cpu().numpy()
            true.append(a)
            predicted.append(b)
            truepositive.update(tp)
            truenegative.update(tn)
            falsepositive.update(fp)
            falsenegative.update(fn)
            accuracy.update(acc)
            #logger('Epoch\tAccuracy\tAverageAccuracy\tTruePositive\tTrueNegative\tFalsePositive\tFalseNegative\tPath','info','testlog')
            logger('{0}\t{ac.val:.2f}\t{ac.avg:.6f}\t{tp.sum:.2f}\t{tn.sum:.2f}\t{fp.sum:.2f}\t{fn.sum:.2f}\t{p}'
                   .format(epoch,ac=accuracy,tp=truepositive,tn=truenegative,
                           fp=falsepositive,fn=falsenegative,p=path),'info','testlog')
            
            niter = epoch*len(testLoader)+i
            #writer.add_scalar('Test/Loss', losses.val, niter)
            #writer.add_scalar('Test/Prec', avgaccu.val, niter)
            #writer.add_scalar('Test/AvgLoss', losses.avg, niter)
            writer.add_scalar('Test/AvgPrec', accuracy.avg, niter)




def train(train_loader,model,criterion,optimizer,epoch):
    
    model.train()

    end = time.time()
    for i,(batch,bag_label,_) in enumerate(train_loader):
        data_time.update(time.time()-end)
        images = batch[0].to(device)
        labels = bag_label[0].to(device)

        
        scores,pred = model(images)
        loss = criterion(scores,labels)
        
        # top-k ? accuaracy 
        # for now evaluating normal accuracy
        acc = accuracyperImage(pred,labels)
        
        #loss.item() to get the loss value from loss tensor
        losses.update(loss.item(), images.size(0))
        avgaccu.update(acc,images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        logger('[{0}][{1}/{2}]\t{batch_time.val:.6f}\t {batch_time.avg:.6f}\t'
              '{loss.val:.9f}\t{loss.avg:.9f}\t'
              '{acc.val:.9f}\t{acc.avg:.9f}\t'.format(epoch,i,len(train_loader),batch_time=batch_time,
                                                      loss=losses, acc=avgaccu),'info','trainlog')
        niter = epoch*len(train_loader)+i
        writer.add_scalar('Train/Loss', losses.val, niter)
        #writer.add_scalar('Train/Prec', avgaccu.val, niter)
        writer.add_scalar('Train/AvgLoss', losses.avg, niter)
        writer.add_scalar('Train/AvgPrec', avgaccu.avg, niter)


# In[23]:


def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avgaccu =  AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (batch, bag_label,_) in enumerate(val_loader):
            input= batch[0].to(device)
            target = bag_label[0].to(device)
            
            # compute output
            scores,pred = model(input)
            loss = criterion(scores, target)

            # measure accuracy and record loss
            acc= accuracyperImage(pred, target)
            losses.update(loss.item(), input.size(0))
            avgaccu.update(acc,input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger('[{0}/{1}]\t'
                  '{batch_time.val:.6f}\t{batch_time.avg:.6f}\t'
                  '{loss.val:.9f}\t{loss.avg:.9f}\t'
                  '{acc.val:.9f}\t{acc.avg:.9f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,acc=avgaccu),'info','vadLog')
            niter = epoch*len(val_loader)+i
            #writer.add_scalar('Val/Loss', losses.val, niter)
            #writer.add_scalar('Val/Prec', avgaccu.val, niter)
            writer.add_scalar('Val/AvgLoss', losses.avg, niter)
            writer.add_scalar('Val/AvgPrec', avgaccu.avg, niter)            
            
            
        print(' * Acc {acc.avg:.9f}'
              .format(acc=avgaccu))
    


    return avgaccu.avg



# # Logger

# In[11]:


trainLogFile = openfile('./logs/'+experimentName+'/train.tsv')
valLogFile = openfile('./logs/'+experimentName+'/validation.tsv')
testLogFile = openfile('./logs/'+experimentName+'/test.tsv')
setup_logger('trainlog',trainLogFile)
setup_logger('testlog',testLogFile)
setup_logger('vadLog',valLogFile)
basiclogging('trainlog')
basiclogging('testlog')
basiclogging('vadLog')


batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
avgaccu =  AverageMeter()

# # Training Loop

# In[ ]:

for epoch in range(start_epoch , numEpochs):
       
    adjust_learning_rate(optimizer,epoch,lr)
    
    
    train(trainLoader,model,criterion,optimizer,epoch)
    
    
    
    acc =  validate(valLoader,model,criterion,epoch)
    
    
    
    is_best = acc > best_acc
    
    best_acc = max(acc,best_acc)
    
    #is_best=False
    #saving the checkpoint if is_best is True
    save_checkpoint({
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_acc':best_acc,
        'optimizer':optimizer.state_dict(),
    },is_best)
    
    if(is_best):
        test(testLoader,model,epoch)
    


# In[ ]:




