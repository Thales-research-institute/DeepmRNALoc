import imp
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pickle
import os
# 读取数据 shuffle
from h5py import File
def load_data(root_path):
    with File(root_path + os.sep + 'Data/H5/mRNA_kmer_k=12345678.h5','r') as f:
        x_train_12345 = f['kmer_x_train_all'].value
        y_train_12345 = f['kmer_y_train_all'].value
        x_test_12345 = f['kmer_x_test'].value
        y_test_12345 = f['kmer_y_test'].value

        
    print(x_train_12345.shape)
    print(x_test_12345.shape)


    with File(root_path + os.sep + 'Data/H5/mRNA_CGRS_cut.h5','r') as f:
        x_train_CGR = f['CGR_x_train_all'].value
        y_train_CGR = f['CGR_y_train_all'].value
        x_test_CGR = f['CGR_x_test'].value
        y_test_CGR = f['CGR_y_test'].value


    print(x_train_CGR.shape)
    print(x_test_CGR.shape)
    x_train_CGR = x_train_CGR / 255.0
    x_test_CGR = x_test_CGR / 255.0


    x_train_CGR = x_train_CGR.reshape(-1,184*247)
    x_test_CGR = x_test_CGR.reshape(-1,184*247)
    x_train_12345 = np.squeeze(x_train_12345)
    x_test_12345 = np.squeeze(x_test_12345)

    print(x_train_12345.shape)
    print(x_test_12345.shape)
    print(x_train_CGR.shape)
    print(x_test_CGR.shape)

    new_x_train_all = np.concatenate((x_train_12345,x_train_CGR),axis=1)
    new_y_train_all = y_train_CGR
    new_x_test_all = np.concatenate((x_test_12345,x_test_CGR),axis=1)
    new_y_test_all = y_test_CGR

    x_all = np.concatenate((new_x_train_all,new_x_test_all),axis = 0)
    y_all = np.concatenate((new_y_train_all,new_y_test_all),axis = 0)

    scaler = preprocessing.StandardScaler()

    x_train2,x_test2,y_train2,y_test2 = train_test_split(x_all, y_all, random_state = 11,test_size=2499,stratify=y_all)

    f = open(root_path + os.sep + 'checkpoints/scalar.pkl','wb')
    x_train2 = scaler.fit_transform(x_train2)
    x_test2 = scaler.transform(x_test2)
    pickle.dump(scaler, f)
    return x_train2,x_test2,y_train2,y_test2
