#!/usr/bin/env python
# coding: utf-8

import os
import sys

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import matplotlib.pyplot as plt
from pathlib import Path
import CDDSM
from tqdm import tqdm

#Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 500
batch_size = 1
learning_rate = 0.01
num_classes = 2
patch_size =H=W=512


def pause(strg):
    if(strg!=''):
        print('Reached at {}, Press any key to continue'.format(strg))
    else:
        print('Paused, Press any to continue')
    input()
    return


homedir = str(Path.home())
print(homedir)

# train_df = CDDSM.createTrainFrame(homedir)
# test_df = CDDSM.createTestFrame(homedir)
# mammogram_dir = '/home/himanshu/CuratedDDSM/'
# train_file = mammogram_dir+'train.csv'
# test_file = mammogram_dir+'test.csv'
# train_df.to_csv(train_file)
# test_df.to_csv(test_file)

# classes = ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')
# Created a cleaned data file in train.csv and test.csv

train_file = 'train.csv'
test_file = 'test.csv'

# Making of CBIS-DDSM Dataset (train,val,test)

dataset =  CDDSM.MammographyDataset(train_file,homedir,patch_size)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,patch_size)

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

# Length of each Dataset

numberOfTrainData = train_dataset.__len__()
numberOfValData = val_dataset.__len__()
numberOfTestData =  test_dataset.__len__()

total_step=len(train_loader)

print('Size of training dataset {}'.format(numberOfTrainData))
print('Size of Validation dataset {}'.format(numberOfValData))
print('Size of testing dataset {}'.format(numberOfTestData))
print('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))


import torchvision.models as models
resnet = models.resnet18(pretrained=True)


# removing last layer of resnet and grad false

resnet = nn.Sequential(*list(resnet.children())[:-1])


for param in resnet.parameters():
    param.requires_grad = False
    
    
class myCustomModel(torch.nn.Module):
    def __init__(self,pretrainedModel):
        super(myCustomModel,self).__init__()
        
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
        return features , x
        

def getCustomPretrained(model):
    return myCustomModel(model)
    
    
class MIL_loss(torch.nn.Module):
    ''' MIL Loss Layer'''
    def __init__(self,lamda):
        super(MIL_loss, self).__init__()
        self.lamda = lamda
        
    def forward(self,scores,labels):
        ''' lamda is postive constant,
        to convert scores(h_i) into probability'''
        lamda = self.lamda
        scores =  F.relu(scores)
        scores = -(lamda * scores)
        prob_of_bag_not_class = torch.prod(torch.exp(scores),dim=0)
        probBag = 1 - prob_of_bag_not_class
        sum_scores = torch.sum(scores,dim=0)
        loss = -torch.dot(1-labels.float(),sum_scores.float())
        return loss,probBag

# def prediction(scores,labels,lamda):
#     scores =  scores.data
# In[ ]:


# model = B.getModel(3).to(device)
model=getCustomPretrained(resnet)
model=model.to(device)

# store best prediction in one epoch

best_prec = 0

# criterion = nn.CrossEntropyLoss()
criterion = MIL_loss(0.5)
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)


correct=0
# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (i+1) % 50 == 0:
        logging.info ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    if (epoch % 5 ==0):

        previouscorrect = correct

        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)


        logging.info("\n *********************Testing on CBIS-DDSM test data************************\n")

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # logging.info('Correct Label is : {}, Predicted Label is {}'.format(labels,predicted))

            logging.info('Test Accuracy of the model on the {} test images: {} %'.format(number_of_testing_data,100 * correct / total))
            logging.info(' Correct {} Previous correct {}'.format(correct,previouscorrect))
            if(previouscorrect<correct):
                logging.info('Saving the model')
                torch.save(model.state_dict(), './models/Run'+suffix+'model.ckpt')
