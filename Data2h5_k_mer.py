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

# classnames = ['Cytoplasm','Endoplasmic_reticulum','Extracellular_region'.'Mitochondria'.'Nucleus']
classnamemap = {'Cytoplasm':0,'Endoplasmic_reticulum':1,'Extracellular_region':2,'Mitochondria':3,'Nucleus':4}
# 代码有问题 先暂时用这个


x_train_all_k1 = []
y_train_all_k1 = []
x_test_k1 = []
y_test_k1 = []

path_k_mer = '.\\Data\\k_mer\\k=1'
print("k=1")


path_list = glob.glob(path_k_mer+'\\*')

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
        csv = pd.read_csv(x2)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all_k1.append(csv.values[:,1:])
            y_train_all_k1.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test_k1.append(csv.values[:,1:])
            y_test_k1.append(classnamemap[classname])



x_train_all_k2 = []
y_train_all_k2 = []
x_test_k2 = []
y_test_k2 = []

path_k_mer = '.\\Data\\k_mer\\k=2'
print("k=2")

path_list = glob.glob(path_k_mer+'\\*')

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
        csv = pd.read_csv(x2)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all_k2.append(csv.values[:,1:])
            y_train_all_k2.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test_k2.append(csv.values[:,1:])
            y_test_k2.append(classnamemap[classname])


x_train_all_k3 = []
y_train_all_k3 = []
x_test_k3 = []
y_test_k3 = []

path_k_mer = '.\\Data\\k_mer\\k=3'
print("k=3")


path_list = glob.glob(path_k_mer+'\\*')

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
        csv = pd.read_csv(x2)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all_k3.append(csv.values[:,1:])
            y_train_all_k3.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test_k3.append(csv.values[:,1:])
            y_test_k3.append(classnamemap[classname])




x_train_all_k4 = []
y_train_all_k4 = []
x_test_k4 = []
y_test_k4 = []

path_k_mer = '.\\Data\\k_mer\\k=4'
print("k=4")


path_list = glob.glob(path_k_mer+'\\*')

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
        csv = pd.read_csv(x2)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all_k4.append(csv.values[:,1:])
            y_train_all_k4.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test_k4.append(csv.values[:,1:])
            y_test_k4.append(classnamemap[classname])



x_train_all_k5 = []
y_train_all_k5 = []
x_test_k5 = []
y_test_k5 = []

path_k_mer = '.\\Data\\k_mer\\k=5'
print("k=5")


path_list = glob.glob(path_k_mer+'\\*')

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
        csv = pd.read_csv(x2)
        # print(csv)
        # print(csv.shape)
        if trainortest == 'train':
            x_train_all_k5.append(csv.values[:,1:])
            y_train_all_k5.append(classnamemap[classname])
        elif trainortest == 'test':
            x_test_k5.append(csv.values[:,1:])
            y_test_k5.append(classnamemap[classname])

x_train_all_k1 = np.array(x_train_all_k1)
x_train_all_k2 = np.array(x_train_all_k2)
x_train_all_k3 = np.array(x_train_all_k3)
x_train_all_k4 = np.array(x_train_all_k4)
x_train_all_k5 = np.array(x_train_all_k5)

print(x_train_all_k1.shape)
print(x_train_all_k2.shape)
print(x_train_all_k3.shape)
print(x_train_all_k4.shape)
print(x_train_all_k5.shape)

x_train_all = np.concatenate((x_train_all_k1,x_train_all_k2),axis=1)
x_train_all = np.concatenate((x_train_all,x_train_all_k3),axis=1)
x_train_all = np.concatenate((x_train_all,x_train_all_k4),axis=1)
x_train_all = np.concatenate((x_train_all,x_train_all_k5),axis=1)

print(x_train_all.shape)


x_test_all_k1 = np.array(x_test_k1)
x_test_all_k2 = np.array(x_test_k2)
x_test_all_k3 = np.array(x_test_k3)
x_test_all_k4 = np.array(x_test_k4)
x_test_all_k5 = np.array(x_test_k5)

print(x_test_all_k1.shape)
print(x_test_all_k2.shape)
print(x_test_all_k3.shape)
print(x_test_all_k4.shape)
print(x_test_all_k5.shape)

x_test_all = np.concatenate((x_test_all_k1,x_test_all_k2),axis=1)
x_test_all = np.concatenate((x_test_all,x_test_all_k3),axis=1)
x_test_all = np.concatenate((x_test_all,x_test_all_k4),axis=1)
x_test_all = np.concatenate((x_test_all,x_test_all_k5),axis=1)

print(x_test_all.shape)


y_train_all = y_train_all_k1
y_test_all = y_test_k1

 
import h5py
import numpy as np
with h5py.File(".\\Data\\collections\\mRNA_k_mer_k=12345.h5", 'w') as f:
    f.create_dataset('mRNA_included',data=np.array(['Cytoplasm'.encode(),'Endoplasmic_reticulum'.encode(),'Extracellular_region'.encode(),'Mitochondria'.encode(),'Nucleus'.encode()]))
    f.create_dataset('k_mer_x_train_all',data = x_train_all)
    f.create_dataset('k_mer_y_train_all',data = y_train_all)
    f.create_dataset('k_mer_x_test',data = x_test_all)
    f.create_dataset('k_mer_y_test',data = y_test_all)