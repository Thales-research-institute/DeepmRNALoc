# k_mer feature extraction
import numpy as np
import pandas as pd
from get_k_mer import get_kmer
import glob


def fasta2k_mer(filename,filename_result):
    path_list = glob.glob(filename+'/*')
    # print(path_list[:10])
    column_names = [0]

    for path in path_list:
        seq = get_kmer(path,k)
        # print(seq)
        # names = path.split('/')
        names = path.split('\\')
        print(names[-1])
        name = names[-1][:-6]
        print(name)
        test=pd.DataFrame(columns=column_names,data=seq)
        test.to_csv(filename_result+'/'+name+'.txt',encoding='gbk')

for k in range(1,6):
    #Cytoplasm_train
    filename = './Data/fasta/Cytoplasm_train'
    filename_result = './Data/k_mer/k='+str(k)+'/Cytoplasm_train'
    fasta2k_mer(filename,filename_result)

    # Cytoplasm_test
    filename = './Data/fasta/Cytoplasm_test'
    filename_result = './Data/k_mer/k='+str(k)+'/Cytoplasm_test'
    fasta2k_mer(filename,filename_result)

    # Endoplasmic_reticulum_train
    filename = './Data/fasta/Endoplasmic_reticulum_train'
    filename_result = './Data/k_mer/k='+str(k)+'/Endoplasmic_reticulum_train'
    fasta2k_mer(filename,filename_result)

    # Endoplasmic_reticulum_test
    filename = './Data/fasta/Endoplasmic_reticulum_test'
    filename_result = './Data/k_mer/k='+str(k)+'/Endoplasmic_reticulum_test'
    fasta2k_mer(filename,filename_result)

    # Extracellular_region_train
    filename = './Data/fasta/Extracellular_region_train'
    filename_result = './Data/k_mer/k='+str(k)+'/Extracellular_region_train'
    fasta2k_mer(filename,filename_result)

    # Extracellular_region_test
    filename = './Data/fasta/Extracellular_region_test'
    filename_result = './Data/k_mer/k='+str(k)+'/Extracellular_region_test'
    fasta2k_mer(filename,filename_result)

    # Mitochondria_train
    filename = './Data/fasta/Mitochondria_train'
    filename_result = './Data/k_mer/k='+str(k)+'/Mitochondria_train'
    fasta2k_mer(filename,filename_result)

    # Mitochondria_test
    filename = './Data/fasta/Mitochondria_test'
    filename_result = './Data/k_mer/k='+str(k)+'/Mitochondria_test'
    fasta2k_mer(filename,filename_result)

    # Nucleus_train
    filename = './Data/fasta/Nucleus_train'
    filename_result = './Data/k_mer/k='+str(k)+'/Nucleus_train'
    fasta2k_mer(filename,filename_result)

    # Nucleus_test
    filename = './Data/fasta/Nucleus_test'
    filename_result = './Data/k_mer/k='+str(k)+'/Nucleus_test'
    fasta2k_mer(filename,filename_result)