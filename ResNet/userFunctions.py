import os
import torch
import shutil
import time
import torch.nn as nn
import torchvision.models as models



def pause(strg):
    if(strg!=''):
        print('Reached at {}, Press any key to continue'.format(strg))
    else:
        print('Paused, Press any to continue')
    input()
    return



