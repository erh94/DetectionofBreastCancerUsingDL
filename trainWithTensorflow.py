import os
import sys
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from pathlib import Path
from TrainFunctions import *

import BaseModel as B
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs',comment="baseline")


# import file where CBIS dataset
import CDDSM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
num_classes = 3
batch_size = 5
learning_rate = 0.0001

total_iteration = 10000

homedir = str(Path.home())
homedir

train_df = CDDSM.createTrainFrame(homedir)
test_df = CDDSM.createTestFrame(homedir)
mammogram_dir = '/home/himanshu/CuratedDDSM/'
train_file = mammogram_dir+'train.csv'
test_file = mammogram_dir+'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)

classes = ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')

dataset =  CDDSM.MammographyDataset(train_file,homedir,img_resize)
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize)

train_dataset , val_dataset = CDDSM.trainValSplit(dataset,val_share=0.9)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
                                          
numberOfTrainData = train_dataset.__len__()
numberOfValData = val_dataset.__len__()
numberOfTestData =  test_dataset.__len__()

total_step=len(train_loader)

print('Size of training dataset {}'.format(numberOfTrainData))
print('Size of Validation dataset {}'.format(numberOfValData))
print('Size of testing dataset {}'.format(numberOfTestData))
print('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,learning_rate,H,W,total_step))


model = B.getModel1024L(3).to(device)

# store best prrediction in one epoch

best_prec = 0


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


for epoch in range(start_epoch , epochs):

    adjust_learning_rate(optimizer,epoch)

    # train for one epoch
    train(train_loader,model,criterion,optimizer,epoch)

    # evalaute on validation set
    prec = validate(val_loader,model,criterion)


    # to save best model after each epoch
    is_best = prec > best_prec
    best_prec =  max(prec, best_prec)
    save_checkpoint({
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_prediction':best_prec,
        'optimizer':optimizer.state_dict(),
    },is_best)


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
