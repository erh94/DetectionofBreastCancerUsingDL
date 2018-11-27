import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path
from customDataset import *
from tqdm import tqdm
from utilsFn import *
import time
import copy
from collections import Counter

experimentName = time.strftime("%d%b%Y%H%M",time.localtime())
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/{}'.format(experimentName))



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Patch Classifier Training')

#parser.add_argument('data', metavar='DIR',
#                   help='path to dataset')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')


parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

############################ global variables ###############################
args = parser.parse_args()

num_epochs = args.epochs

batch_size = args.batch_size

learning_rate = args.lr

imageSize = 512

best_acc = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

traindir = './images/train/'
testdir = './images/test/'

experimentName='ResNet18'

prefix = 'Epochs{}Batch{}Image{}lr{}'.format(num_epochs,batch_size,imageSize,learning_rate)    
########################### custom resnet for 512 * 512 grayscale images ###################### 
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
    

def main():
    global args, best_acc,device,traindir,testdir
    
    


    dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                                                transforms.Grayscale(1),
                                                                torchvision.transforms.RandomHorizontalFlip(),
                                                                torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                                                transforms.ToTensor()
                                                                ]),)
    train_dataset , val_dataset = trainValSplit(dataset,val_share=0.1)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size, 
                                           shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=args.batch_size, 
                                           shuffle=True)
        
    total_size = dataset.__len__()
    numberOfTrainData = train_dataset.__len__()
    numberOfValData = val_dataset.__len__()
    # numberOfTestData =  test_dataset.__len__()


    ############# determining size of each class ##################
    num_classes = len(dataset.classes)
    print('Number of classes : {}'.format(num_classes))
    print('Classes : {}'.format(dataset.classes))
    (x,y) = dataset[0]
    print(x.shape,y)
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
    class_counts = {}
    class_counts = dict(Counter(sample_tup[1] for sample_tup in dataset.imgs))
    print(class_counts[0])
    print(class_counts[1])
    print(class_counts[2])


    # class_counts['val'] = dict(Counter(sample_tup[1] for sample_tup in dataset.imgs))
    class_weights = [1-(float(class_counts[class_id])/total_size) for class_id in range(num_classes)]
    print(class_weights)
    




    total_step=len(train_loader)
    H=W=imageSize
    print('Size of training dataset {}'.format(numberOfTrainData))
    print('Size of Validation dataset {}'.format(numberOfValData))
    # print('Size of testing dataset {}'.format(numberOfTestData))
    print('No. of Epochs: {}\n Batch size: {}\n Learning_rate : {}\n Image size {}*{}\n Step {}'
            .format(num_epochs,batch_size,learning_rate,H,W,total_step))
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    resnet = models.resnet18(pretrained=True)
    features_len = resnet.fc.in_features
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    pause('Start training?')

    
    # print(x,y)

    # assert False,'check'

    model=getCustomPretrained(resnet,features_len,num_classes)
    model=model.to(device)

    # print(model)

    # for param in resnet.parameters():
    #     param.requires_grad = False



    # store best prediction in one epoch

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,weight_decay=args.weight_decay)


    model.layer0.apply(init_weights)

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



    
    if args.evaluate:
        validate(val_loader, model, criterion,1)
        return


    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch,learning_rate)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_acc
        best_prec1 = max(prec1, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

###################### SAVE CHECKPOINT ############################



def save_checkpoint(state,is_best,filename=openfile('./models/{}/checkpoint{}.pth.tar'.format(experimentName,prefix))):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,openfile('./models/{}/best_{}.pth.tar'.format(experimentName,prefix)))


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
        # print(images.shape)
        #output = model(images)
        _,output = model(images)
        # print(output)
        # print(labels)
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

        print('Epoch [{0}][{1}/{2}],'
              'BatchTime {batch_time.val:.3f},{batch_time.avg:.3f},'
              'Loss : {loss.val:.4f},{loss.avg:.4f},'
              'Accuracy : {acc.val:.4f},{acc.avg:.4f}'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, acc=avgaccu),'info','trainlog')

        niter = epoch*len(train_loader)+i
        writer.add_scalar('Train/Loss', losses.val, niter)
        writer.add_scalar('Train/Prec', avgaccu.val, niter)



def validate(val_loader, model, criterion,epoch):
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
            print('Validation')
            print('[{0}/{1}],'
                  '{batch_time.val:.3f},{batch_time.avg:.3f},'
                  '{loss.val:.4f},{loss.avg:.4f}),'
                  '{acc1.val:.3f},{acc1.avg:.3f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,acc1=avgaccu),'info','vadLog')

        print(' **********Validation Accuracy {acc1.avg:.3f}*****************'
              .format(acc1=avgaccu))


        niter = epoch*len(val_loader)+i
        writer.add_scalar('Val/Loss', losses.val, niter)
        writer.add_scalar('Val/Acc', avgaccu.val, niter)


    return avgaccu.val


                                                
if __name__ == '__main__':
    main()

