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
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder




# # In[2]:


# import warnings
# warnings.filterwarnings("ignore")
# plt.ion()



def createTrainFrame(homedir):
    
    train_mass_csv = pd.read_csv(homedir+"/CuratedDDSM/Train/mass_case_description_train_set.csv")
    train_calc_csv = pd.read_csv(homedir+"/CuratedDDSM/Train/calc_case_description_train_set.csv")
    train_calc_csv = train_calc_csv.rename(columns={'breast density': 'breast_density'})
    train_calc_csv['image file path'] = 'Calc/CBIS-DDSM/'+ train_calc_csv['image file path']
    train_calc_csv['ROI mask file path'] ='CalcROI/CBIS-DDSM/' + train_calc_csv['ROI mask file path']
    train_calc_csv['cropped image file path'] ='CalcROI/CBIS-DDSM/' + train_calc_csv['cropped image file path']
    train_mass_csv['image file path'] = 'Mass/CBIS-DDSM/' + train_mass_csv['image file path']
    train_mass_csv['ROI mask file path'] ='MassROI/CBIS-DDSM/' + train_mass_csv['ROI mask file path']
    train_mass_csv['cropped image file path'] ='MassROI/CBIS-DDSM/'+ train_mass_csv['cropped image file path']
    common_col = list(set(train_calc_csv.columns) & set(train_mass_csv.columns))
    train = pd.concat([train_mass_csv[common_col],train_calc_csv[common_col]], ignore_index=True,sort='False')
    #train = train_mass_csv
    train['image file path'] = 'CuratedDDSM/Train/'+train['image file path'] 
    train['ROI mask file path'] = 'CuratedDDSM/Train/'+train['ROI mask file path']
    train['cropped image file path'] = 'CuratedDDSM/Train/'+train['cropped image file path']
    train['pathology_class'] = LabelEncoder().fit_transform(train['pathology'])
    return train

def createTestFrame(homedir):
    test_mass_csv = pd.read_csv(homedir+"/CuratedDDSM/Test/mass_case_description_test_set.csv")
    test_calc_csv = pd.read_csv(homedir+"/CuratedDDSM/Test/calc_case_description_test_set.csv")
    test_calc_csv = test_calc_csv.rename(columns={'breast density': 'breast_density'})
    test_calc_csv['image file path'] = 'Calc/CBIS-DDSM/'+ test_calc_csv['image file path']
    test_calc_csv['ROI mask file path'] ='CalcROI/CBIS-DDSM/' + test_calc_csv['ROI mask file path']
    test_calc_csv['cropped image file path'] ='CalcROI/CBIS-DDSM/' + test_calc_csv['cropped image file path']
    test_mass_csv['image file path'] = 'Mass/CBIS-DDSM/' + test_mass_csv['image file path']
    test_mass_csv['ROI mask file path'] ='MassROI/CBIS-DDSM/' + test_mass_csv['ROI mask file path']
    test_mass_csv['cropped image file path'] ='MassROI/CBIS-DDSM/'+ test_mass_csv['cropped image file path']
    common_col = list(set(test_calc_csv.columns) & set(test_mass_csv.columns))
    test = pd.concat([test_mass_csv[common_col], test_calc_csv[common_col]], ignore_index=True,sort='False')
    test['image file path'] = 'CuratedDDSM/Test/'+test['image file path'] 
    test['ROI mask file path'] = 'CuratedDDSM/Test/'+test['ROI mask file path']
    test['cropped image file path'] = 'CuratedDDSM/Test/'+test['cropped image file path']
    test['pathology_class'] = LabelEncoder().fit_transform(test['pathology'])

    return test


# In[6]:


class MammographyDataset(Dataset):
    """Creating CBIS-DDSM pytorch dataset."""

    def __init__(self, csv_file, root_dir,img_size):
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
        # Image Columns
        self.image_arr = np.asarray(self.frame['image file path'])
        # Labels
        self.label_arr = np.asarray(self.frame['pathology_class'])
        # Calculate Len
        self.data_len = len(self.frame.index)
        # Location of Curated DDSM
        self.root_dir = root_dir
        # Transformations to convert image into 512*512
        self.transformations =             transforms.Compose([transforms.Grayscale(1),
                               transforms.Resize(img_size,interpolation=Image.LANCZOS),
                               transforms.CenterCrop(img_size)])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Open image
        #img_as_img = Image.open(single_image_name)
        image_path = os.path.join(self.root_dir,self.image_arr[index])
        image_dcm= DCM.read_file(image_path)
        
        image_2d = image_dcm.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0)/ image_2d.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        #Transform image to 512 * 512 
        img = Image.fromarray(image_2d_scaled)
        img_as_img = self.transformations(img)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        image_label = self.label_arr[index]

        return (img_as_tensor, image_label)



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



