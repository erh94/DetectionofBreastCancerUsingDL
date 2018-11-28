import os
import torch
import shutil
import time
import torch.nn as nn
import torchvision.models as models
import logging
from pathlib import Path



def pause(strg):
    if(strg!=''):
        print('Reached at {}, Press any key to continue'.format(strg))
    else:
        print('Paused, Press any to continue')
    input()
    return

def init_weights(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def logger(msg, level, logfile):
    log=logging.getLogger(logfile)
    if level == 'info'    : log.info(msg) 
    if level == 'warning' : log.warning(msg)
    if level == 'error'   : log.error(msg)
        
def setup_logger(logger_name, log_file, level=logging.INFO):

    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)

def openfile(filename):
    path = Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    return filename



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



def adjust_learning_rate(optimizer,epoch,initLR):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epoch'''
    lr = initLR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def accuracy(output,target):
    with torch.no_grad():
        batch_size =  target.size(0)
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        #print('Predicted:{}\nTrue:{}'.format(predicted.cpu().detach(),target.cpu().detach()))
        #pause('h')
        correct = (predicted == target).sum().item()
        acc = correct/total
    return acc





