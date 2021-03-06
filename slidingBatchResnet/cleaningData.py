

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
import progressbar as pb

i=0

def createTrainFrame(homedir):

    widgets = ['Progress for train data cleaning: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

    
    
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
    
    train['image file path'] = homedir+'/CuratedDDSM/Train/'+train['image file path'] 
    train['ROI mask file path'] = homedir+'/CuratedDDSM/Train/'+train['ROI mask file path']
    train['cropped image file path'] = homedir+'/CuratedDDSM/Train/'+train['cropped image file path']
    
    train=train[train.pathology!='BENIGN_WITHOUT_CALLBACK']
    
    train['pathology_class'] = LabelEncoder().fit_transform(train['pathology'])
    train =  pd.concat([train,pd.get_dummies(train['pathology'])],axis=1)
    
    ''' proper ROI mask present check'''
    
    train['image file path']=train['image file path'].str.replace('\n',"")
    train['ROI mask file path']=train['ROI mask file path'].str.replace('\n',"")
    train['cropped image file path']=train['cropped image file path'].str.replace('\n','')
    
    train = train.reset_index(drop=True)
    timer = pb.ProgressBar(widgets=widgets, maxval=len(train)+1).start()    
    notfound = []
    for index, row in train.iterrows():
        timer.update(index)
        if(os.path.isfile(row['cropped image file path'])==False
           or os.path.isfile(row['ROI mask file path'])==False or os.path.isfile(row['image file path'])==False):
            notfound.append(index)
    
    # print(index)
    # print(len(train))
    timer.finish()
    
    
    
    train.drop(train.index[notfound], inplace=True)
    
    train = train.reset_index(drop=True)
    
    


    okcount=0
    count=0
    deleteIndex = []
    timer = pb.ProgressBar(widgets=widgets, maxval=len(train)+1).start()
    for index, row in train.iterrows():
        mask = DCM.dcmread(row['ROI mask file path']).pixel_array 
        crop = DCM.dcmread(row['cropped image file path']).pixel_array
        img = DCM.dcmread(row['image file path']).pixel_array
        if(img.shape==mask.shape):
            okcount=okcount+1
        else:
            if(img.shape==crop.shape):
                okcount = okcount+1
                train.loc[index,['cropped image file path','ROI mask file path']] = train.loc[index,['ROI mask file path','cropped image file path']].values
            else:
                deleteIndex.append(index)
        timer.update(index)
    
    timer.finish()

    train.drop(train.index[deleteIndex], inplace=True)
    
    train = train.reset_index(drop=True)    
    train.to_csv("train.csv")
    
    
    return train

def createTestFrame(homedir):

    widgets = ['Progress for test data cleaning: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

    
    
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
    test['image file path'] = homedir+'/CuratedDDSM/Test/'+test['image file path'] 
    test['ROI mask file path'] = homedir+'/CuratedDDSM/Test/'+test['ROI mask file path']

    test=test[test.pathology!='BENIGN_WITHOUT_CALLBACK']

    test['cropped image file path'] = homedir+'/CuratedDDSM/Test/'+test['cropped image file path']
    test['pathology_class'] = LabelEncoder().fit_transform(test['pathology'])
    
    test =  pd.concat([test,pd.get_dummies(test['pathology'])],axis=1)
    
    ''' proper ROI mask present check'''
    
    test['image file path']=test['image file path'].str.replace('\n',"")
    test['ROI mask file path']=test['ROI mask file path'].str.replace('\n',"")
    test['cropped image file path']=test['cropped image file path'].str.replace('\n','')
    test = test.reset_index(drop=True)
    
    timer = pb.ProgressBar(widgets=widgets, maxval=len(test)+1).start()
    notfound = []
    for index, row in test.iterrows():
        if(os.path.isfile(row['cropped image file path'])==False
           or os.path.isfile(row['ROI mask file path'])==False or os.path.isfile(row['image file path'])==False):
            notfound.append(index)
        timer.update(index)
    
    timer.finish()
    
    test.drop(test.index[notfound], inplace=True)
    
    test = test.reset_index(drop=True)
    
    
    okcount=0
    count=0
    deleteIndex = []
    timer = pb.ProgressBar(widgets=widgets, maxval=len(test)+1).start()

    for index, row in test.iterrows():
        timer.update(index)
        mask = DCM.dcmread(row['ROI mask file path']).pixel_array 
        crop = DCM.dcmread(row['cropped image file path']).pixel_array
        img = DCM.dcmread(row['image file path']).pixel_array
        if(img.shape==mask.shape):
            okcount=okcount+1
        else:
            if(img.shape==crop.shape):
                okcount = okcount+1
                test.loc[index,['cropped image file path','ROI mask file path']] = test.loc[index,['ROI mask file path','cropped image file path']].values
            else:
                deleteIndex.append(index)
    
    timer.finish()
    test.drop(test.index[deleteIndex], inplace=True)
    
    test = test.reset_index(drop=True)    
    
    test.to_csv("test.csv")
    
    
    return test


# In[7]:


train = createTrainFrame('/media/erh/BEFCFF6AFCFF1B7B/dataset/')
test = createTestFrame('/media/erh/BEFCFF6AFCFF1B7B/dataset/')
print("Finished")


