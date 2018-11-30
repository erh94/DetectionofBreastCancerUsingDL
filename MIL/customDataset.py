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
import torchvision
import torch
from sklearn.feature_extraction.image import extract_patches_2d
import PIL
from skimage.util import img_as_bool
from skimage import img_as_float



class MammographyDataset(Dataset):
    """Creating CBIS-DDSM pytorch dataset."""

    def __init__(self, csv_file, root_dir,img_size,mode):
        """
        Args:
            csv_file (string): Path to the csv file containing labels.
            root_dir (string): path to CuratedDDSM directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #Image size
        self.img_size = img_size
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.frame = pd.read_csv(csv_file)
        # Image and mask Columns
        self.image_arr = np.asarray(self.frame['image file path'])
        self.mask_arr = np.asarray(self.frame['ROI mask file path'])
        # Labels
        self.label_arr = np.asarray(self.frame['pathology_class'])
        self.num_classes = self.frame.pathology.nunique()
        # Calculate Len
        self.data_len = len(self.frame.index)
        # Location of Curated DDSM
        self.root_dir = root_dir
        self.mode=mode
        # Transformations to convert image into 512*512 / chabge it to random rotate
        # self.transformations =             transforms.Compose([transforms.Grayscale(1),
        #                        transforms.Resize(img_size,interpolation=Image.LANCZOS),
        #                        transforms.CenterCrop(img_size)])
        self.transforms = torchvision.transforms.Compose([
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
                        ])
        self.dataAugmentation = True

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Open image
        if(self.mode=='test'):
            self.dataAugmentation=False
        #img_as_img = Image.open(single_image_name)
        image_path = os.path.join(self.root_dir,self.image_arr[index])
        mask_path =  os.path.join(self.root_dir,self.mask_arr[index])
        
        img= DCM.read_file(image_path).pixel_array.astype(float)
        mask = DCM.read_file(mask_path).pixel_array.astype(float)  
        
        img = (np.maximum(img,0)/ img.max()) * 255
        img = np.uint8(img)
        mask = (np.maximum(mask,0)/mask.max())
        
        imgPIL = Image.fromarray(img)
        if(self.dataAugmentation == True):
            imgPIL = self.transforms(imgPIL)

        img = np.asarray(imgPIL)

        img = img_as_float(img)
        #patches formation
        
        patches = extract_patches_2d(img, (self.img_size,self.img_size),max_patches=100, random_state=5)
        maskpatches = extract_patches_2d(mask,(self.img_size,self.img_size),max_patches=100,random_state=5)
        # labels=[]
        # for eachmask in maskpatches:
        #     labels.append(img_as_bool(eachmask).any())
        # labels = [int(elem) for elem in labels]
        # labels = np.asarray(labels)

        patches_tensor =  torch.from_numpy(patches)

        patches_tensor = patches_tensor.unsqueeze(1).float()
        # labels_tensor = torch.from_numpy(labels)
        patches_tensor = patches_tensor.repeat(1,3,1,1)
        img_tensor = torch.from_numpy(img) 
        label = np.zeros(self.num_classes)
        label_index = self.label_arr[index]
        label[label_index]=1
        label = torch.from_numpy(label).float()
        img_as_tensor = self.to_tensor(imgPIL)
        
        if(self.dataAugmentation == True):
            return (patches_tensor, label,img_as_tensor)
        else:
            return (patches_tensor, label,image_path)
    



class FullTrainingDataset(Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        # super(FullTrainingDataset, self).init()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i+self.offset]


def trainValSplit(dataset, val_share):
    val_offset = int(len(dataset)*(1-val_share))
    
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)



