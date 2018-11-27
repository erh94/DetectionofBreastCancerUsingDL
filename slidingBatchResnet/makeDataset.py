from __future__ import print_function, division
import pydicom
import torch
import os
import sys
import pandas as pd
import numpy as np
import pydicom as DCM
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps, ImageEnhance
from sklearn.preprocessing import LabelEncoder
from skimage import img_as_float
import torchvision
import torch
from sklearn.feature_extraction.image import extract_patches_2d
import PIL
from skimage.util import img_as_bool
import progressbar as pb
from utilsFn import *



import warnings
warnings.filterwarnings("ignore")
plt.ion()

################ Sliding window ###########################

# import the necessary packages
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



''' Making three classes from mammographic images using ROI'''

############ Generating patches from mammographic images training a patch classifier #############

def isbkg(patch):
    numel = patch.shape[0] * patch.shape[1]
    numzeros  =  np.count_nonzero(patch<50)
    if((numzeros/numel)>0.8):
        return True
    return False

train_file = 'train.csv'

####### No need to see this ##############
test_file = 'test.csv'

train_df = pd.read_csv(train_file)

########## Initiliaze Progress Bar ###########

widgets = ['Progress for making dataset: ', pb.Percentage(), ' ',pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

timer = pb.ProgressBar(widgets=widgets, maxval=len(train_df)+1).start()


############ making patches and saving to folder ###################

for idx in range(len(train_df)):
    timer.update(idx)
    filename = '{}_{}_{}_{}'.format(idx,train_df['patient_id'][idx],train_df['image view'][idx],train_df['pathology'][idx])
    imgdcm = train_df['image file path'][idx]
    maskdcm = train_df['ROI mask file path'][idx]
    img =  DCM.read_file(imgdcm).pixel_array.astype(float)
    mask = DCM.read_file(maskdcm).pixel_array.astype(float)
    img = (np.maximum(img,0)/ img.max()) * 255.0
    img = np.uint8(img)
    mask = (np.maximum(mask,0)/mask.max())
    classes = train_df['pathology'][idx]

    #patches = extract_patches_2d(img, (512,512),max_patches=50, random_state=1)
    #maskpatches = extract_patches_2d(mask,(512,512),max_patches=50,random_state=1)
    winSize = 512
    stepSize = winSize/2
    stepSize = int(stepSize)


    patches =  sliding_window(img,stepSize,windowSize=(winSize,winSize))
    maskpatches =sliding_window(mask,stepSize,windowSize=(winSize,winSize))

    ############## Sliding Window patches ################################





    i=0
    for eachmask,eachpatch in zip(maskpatches,patches):
        i+=1
        if(eachmask[2].shape==(512,512)):            

            pil = PIL.Image.fromarray(eachpatch[2])
            if(classes=='MALIGNANT'):
    #             print(eachpatch)
    #             print(eachmask)
    #             print(isbkg(eachpatch))
    #             plt.imshow(pil)
                
    #             assert False,'qwerty'
                if(img_as_bool(eachmask[2]).any()):
                    pil.save(openfile('./images/malignant/{}_{}.png'.format(filename,i)))
                elif(isbkg(eachpatch[2])):
                    #pil.save(openfile('./images/bkg/{}_{}.png'.format(filename,i)))
                    continue
                else:
                    pil.save(openfile('./images/bkg/{}_{}.png'.format(filename,i)))
                
                
            elif(classes=='BENIGN'):
                if(img_as_bool(eachmask[2]).any()):
                    pil.save(openfile('./images/benign/{}_{}.png'.format(filename,i)))
                elif(isbkg(eachpatch[2])):
                    #pil.save(openfile('./images/bkg/{}_{}.png'.format(filename,i)))
                    continue
                else:
                    pil.save(openfile('./images/bkg/{}_{}.png'.format(filename,i)))
            else:
                assert False, 'Unknown class found'
timer.finish()
