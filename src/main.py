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

from torchviz import make_dot
import matplotlib.pyplot as plt
import graphviz

from pathlib import Path
from tqdm import tqdm


# Model Specifics
# import pytorch_resnet as R
import BaseModel as B
import CDDSM
import time
import logging

suffix = time.strftime("%d%b%Y%H%M",time.localtime())
logtime = time.strftime("%d%b%Y %H:%M:%S",time.localtime())

level = logging.INFO
format = '%(message)s'
handlers = [logging.FileHandler('./logs/Run{}'.format(suffix)),logging.StreamHandler()]
logging.basicConfig(level=level,format=format,handlers=handlers)


#Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('Device {}'.format(device))
# Hyper parameters
num_epochs = 150
num_classes = 3
batch_size = 3
learning_rate = 0.0001

# x = torch.randn(batch_size, channels_mammo,heights_mammo , width_mammo)


# In[2]:


# Reading a mammogram
homedir = str(Path.home())
homedir

# CSV preprocessing
train_df = CDDSM.createTrainFrame(homedir)
test_df = CDDSM.createTestFrame(homedir)
mammogram_dir = '/home/himanshu/CuratedDDSM/'
train_file = mammogram_dir+'train.csv'
test_file = mammogram_dir+'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)

# labells = train_df[['pathology','pathology_class']]
# logging.info(labells)


classes = ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')

#Image size
img_resize=H=W=1024

# Mammography dataset
train_dataset =  CDDSM.MammographyDataset(train_file,homedir,img_resize)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

number_of_training_data = train_dataset.__len__()
number_of_testing_data = test_dataset.__len__()


total_step = len(train_loader)


logging.info('Size of training dataset {}'.format(number_of_training_data))
logging.info('Size of testing dataset {}'.format(number_of_testing_data))
logging.info('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))

# In[3]:

model = B.getModel1024L(3).to(device)
# getModel gives a model for images 512*512
# getModel1024 gives model for images 1024*1024
# getModel1024L gives model for images 1024*1024

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



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
