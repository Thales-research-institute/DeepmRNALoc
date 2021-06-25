import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
# import tensorflow as tf
# from tensorflow import keras
import glob
import cv2
# classnames = ['Cytoplasm','Endoplasmic_reticulum','Extracellular_region'.'Mitochondria'.'Nucleus']
classnamemap = {'Cytoplasm':0,'Endoplasmic_reticulum':1,'Extracellular_region':2,'Mitochondria':3,'Nucleus':4}

x_train_all = []
y_train_all = []
x_test = []
y_test = []

path_CGRS = '.\\Data\\CGRS'


path_list = glob.glob(path_CGRS+'\\*')

for x in path_list:
    # print(x)
    folder_name = x.split('\\')[-1]
    print(folder_name)
    labelandtrainortest = folder_name.split('_')
    if len(labelandtrainortest) == 3:
        classname = '_'.join(labelandtrainortest[0:2])
        trainortest = labelandtrainortest[2]
    elif len(labelandtrainortest) == 2:
        classname = labelandtrainortest[0]
        trainortest = labelandtrainortest[1]
    # print(classname)
    # print(trainortest)
    for x2 in glob.glob(x+'\\*'):
        img = cv2.imread(x2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all.append(img)
            y_train_all.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test.append(img)
            y_test.append(classnamemap[classname])



x_train_all = np.array(x_train_all)
y_train_all = np.array(y_train_all)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_train_all.shape)
print(y_train_all.shape)
print(x_test.shape)
print(y_test.shape)


import h5py
import numpy as np
with h5py.File(".\\Data\\collections\\mRNA_CGRS.h5", 'w') as f:
    f.create_dataset('mRNA_included',data=np.array(['Cytoplasm'.encode(),'Endoplasmic_reticulum'.encode(),'Extracellular_region'.encode(),'Mitochondria'.encode(),'Nucleus'.encode()]))
    f.create_dataset('CGR_x_train_all',data = x_train_all)
    f.create_dataset('CGR_y_train_all',data = y_train_all)
    f.create_dataset('CGR_x_test',data = x_test)
    f.create_dataset('CGR_y_test',data = y_test)