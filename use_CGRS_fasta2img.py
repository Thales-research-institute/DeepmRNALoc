# k_mer 特征提取
import numpy as np
import pandas as pd
import glob
import os


command = 'python ./dnacgr.py'
def fastaCGRS(filename,filename_result):
    path_list = glob.glob(filename+'\\*')
    # print(path_list[:10])

    for path in path_list:
        fin_command = command +' '+ path + ' --dest-dir '+ filename_result + ' --save  --dpi 50'
        os.system(fin_command)
        print(path.split('\\')[-1])

# fasta2CGRimg
# Cytoplasm_train
filename = './Data/fasta/Cytoplasm_train'
filename_result = './Data/CGRS/Cytoplasm_train'
fastaCGRS(filename,filename_result)

# Cytoplasm_test
filename = './Data/fasta/Cytoplasm_test'
filename_result = './Data/CGRS/Cytoplasm_test'
fastaCGRS(filename,filename_result)

# Endoplasmic_reticulum_train
filename = './Data/fasta/Endoplasmic_reticulum_train'
filename_result = './Data/CGRS/Endoplasmic_reticulum_train'
fastaCGRS(filename,filename_result)

# Endoplasmic_reticulum_test
filename = './Data/fasta/Endoplasmic_reticulum_test'
filename_result = './Data/CGRS/Endoplasmic_reticulum_test'
fastaCGRS(filename,filename_result)

# Extracellular_region_train
filename = './Data/fasta/Extracellular_region_train'
filename_result = './Data/CGRS/Extracellular_region_train'
fastaCGRS(filename,filename_result)

# Extracellular_region_test
filename = './Data/fasta/Extracellular_region_test'
filename_result = './Data/CGRS/Extracellular_region_test'
fastaCGRS(filename,filename_result)

# Mitochondria_train
filename = './Data/fasta/Mitochondria_train'
filename_result = './Data/CGRS/Mitochondria_train'
fastaCGRS(filename,filename_result)

# Mitochondria_test
filename = './Data/fasta/Mitochondria_test'
filename_result = './Data/CGRS/Mitochondria_test'
fastaCGRS(filename,filename_result)

# Nucleus_train
filename = './Data/fasta/Nucleus_train'
filename_result = './Data/CGRS/Nucleus_train'
fastaCGRS(filename,filename_result)

# Nucleus_test
filename = './Data/fasta/Nucleus_test'
filename_result = './Data/CGRS/Nucleus_test'
fastaCGRS(filename,filename_result)