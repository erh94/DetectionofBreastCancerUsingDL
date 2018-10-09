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


# Model Specifics
# import pytorch_resnet as R
import BaseModel as B
import CDDSM



#Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
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
# print(labells)


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


print('Size of training dataset {}'.format(number_of_training_data))
print('Size of testing dataset {}'.format(number_of_testing_data))
print('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))

# In[3]:

model = B.getModel1024L(3).to(device)
# getModel gives a model for images 512*512
# getModel1024 gives model for images 1024*1024
# getModel1024L gives model for images 1024*1024

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[5]:


# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    torch.save(model.state_dict(), str(str(epoch)+'model.ckpt'))

# Test the model

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)


print("\n *********************Testing on CBIS-DDSM test data************************\n")

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
        print('Correct Label is : {}, Predicted Label is {}'.format(labels,predicted))

    print('Test Accuracy of the model on the {} test images: {} %'.format(number_of_testing_data,100 * correct / total))


torch.save(model.state_dict(), str(str(learning_rate)+'model.ckpt'))
