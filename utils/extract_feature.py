# create folder
import os
import numpy as np
import pandas as pd
import glob

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)           
	else:
		print(path+"folder already exists!")

def create_folder():
    path = './Data'
    mkdir(path)
    path1 = path + '/CGRS'
    mkdir(path1)
    mkdir(path1+'/Cytoplasm_test')
    mkdir(path1+'/Cytoplasm_train')
    mkdir(path1+'/Endoplasmic_reticulum_test')
    mkdir(path1+'/Endoplasmic_reticulum_train')
    mkdir(path1+'/Extracellular_region_test')
    mkdir(path1+'/Extracellular_region_train')
    mkdir(path1+'/Mitochondria_test')
    mkdir(path1+'/Mitochondria_train')
    mkdir(path1+'/Nucleus_test')
    mkdir(path1+'/Nucleus_train')

    for i in range(1,9):
        path2 = path + '/k_mer'
        path2 = path2 + '/k=' + str(i)
        mkdir(path2)
        mkdir(path2+'/Cytoplasm_test')
        mkdir(path2+'/Cytoplasm_train')
        mkdir(path2+'/Endoplasmic_reticulum_test')
        mkdir(path2+'/Endoplasmic_reticulum_train')
        mkdir(path2+'/Extracellular_region_test')
        mkdir(path2+'/Extracellular_region_train')
        mkdir(path2+'/Mitochondria_test')
        mkdir(path2+'/Mitochondria_train')
        mkdir(path2+'/Nucleus_test')
        mkdir(path2+'/Nucleus_train')
    path3 = path + '/fasta'
    mkdir(path3)
    mkdir(path3+'/Cytoplasm_test')
    mkdir(path3+'/Cytoplasm_train')
    mkdir(path3+'/Endoplasmic_reticulum_test')
    mkdir(path3+'/Endoplasmic_reticulum_train')
    mkdir(path3+'/Extracellular_region_test')
    mkdir(path3+'/Extracellular_region_train')
    mkdir(path3+'/Mitochondria_test')
    mkdir(path3+'/Mitochondria_train')
    mkdir(path3+'/Nucleus_test')
    mkdir(path3+'/Nucleus_train')

    path4 = path + '/collections'
    mkdir(path4)


## 使用步骤
### 1、下载文件
### 2、运行create_folder.py
### 3、特征提取
#### 3.1、k_mer特征提取
##### 3.1.1、运行cut_fastas.py
##### 3.1.2、运行k_mer.py
##### 3.1.3、运行Data2h5_k_mer.py
#### 3.2、CGR特征提取
##### 3.2.1、运行use_CGRS_fasta2img.py
##### 3.2.2、运行Data2h5_CGR.py


# cut fastas
def cut_fasta(filename,filename_result):
    files = []
    seq = []
    first_line = ''
    name = ''
    n = 0
    with open(filename) as fs:
        for line in fs:
            # print(line[0])
            if line[0] == '>':
                # n += 1
                if seq != []:
                    fs2 = open(filename_result+'/'+str(name)+'.fasta','w',newline='\n')
                    fs2.writelines(first_line)
                    fs2.writelines(seq)
                name,aft = line.split('#')
                name = name[1:]
                first_line = line
                seq = []
                print(name)
            else:
                seq.append(line)
        
        fs2 = open(filename_result+'/'+str(name)+'.fasta','w',newline='\n')
        fs2.writelines(first_line)
        fs2.writelines(seq)
    # print(n)

def cut_fastas():
    filename = []
    filename_result = []
    # 将一个fasta文件切割为多个
    #Cytoplasm_train
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Cytoplasm_train.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_train')

    # Cytoplasm_test
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Cytoplasm_indep1.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_test')

    # Endoplasmic_reticulum_train
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Endoplasmic_reticulum_train.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_train')

    # Endoplasmic_reticulum_test
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Endoplasmic_reticulum_indep1.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_test')


    # Extracellular_region_train
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Extracellular_region_train.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_train')

    # Extracellular_region_test
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Extracellular_region_indep1.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_test')


    # Mitochondria_train
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Mitochondria_train.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_train')

    # Mitochondria_test
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Mitochondria_indep1.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_test')

    # /home/zshen/workplace/new_mRNA/Data
    # Nucleus_train
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Nucleus_train.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_train')

    # Nucleus_test
    filename.append('/home/zshen/workplace/new_mRNA/Data/data/Nucleus_indep1.fasta')
    filename_result.append('/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_test')



    for i in range(len(filename)):
        print(filename[i])
        print(filename_result[i])
        cut_fasta(filename[i],filename_result[i])


# # kmer
# def get_tris(k):
#     nucle_com = []
#     chars = ['A', 'C', 'G', 'T']
#     base = len(chars)
#     end = len(chars)**k
#     for i in range(0, end):
#         n = i
#         add = ''
#         for j in range(k):
#             ch = chars[n % base]
#             n = int(n/base)
#             add += ch
#         nucle_com.append(add)
#     return nucle_com

# def get_kmer(path,k):
#     fasta = open(path)
#     fasta = fasta.read()
#     sequence = "".join(fasta.split("\n")[1:])
#     sequence = sequence.replace("N", "")
#     print(len(sequence))
#     kmerbases = get_tris(k)

#     kmermap = {}
#     for kmer in  kmerbases:
#         kmermap[kmer] = 0

#     # print(kmermap)
#     for index in range(len(sequence)-k+1):
#         kmermap[sequence[index:index+k]] += 1

#     # print(kmermap)
#     # print(len(kmermap))
#     result = []
#     for kmer in kmermap:
#         result.append(kmermap[kmer])
#     return result


# def fasta2kmer(filename,filename_result):
#     path_list = glob.glob(filename+'/*')
#     # print(path_list[:10])
#     column_names = [0]
#     for path in path_list:
#         seq = get_kmer(path,k)
#         # print(seq)
#         names = path.split('/')
#         print(names[-1])
#         name = names[-1][:-6]
#         # print(name)
#         test=pd.DataFrame(columns=column_names,data=seq)
#         test.to_csv(filename_result+'/'+name+'.txt',encoding='gbk')


# start_k = 1
# end_k = 9
# for k in range(start_k,end_k+1):
#     # 将fasta文件转化问kmer
#     #Cytoplasm_train
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_train'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Cytoplasm_train'
#     fasta2kmer(filename,filename_result)

#     # Cytoplasm_test
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_test'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Cytoplasm_test'
#     fasta2kmer(filename,filename_result)

#     # Endoplasmic_reticulum_train
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_train'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Endoplasmic_reticulum_train'
#     fasta2kmer(filename,filename_result)

#     # Endoplasmic_reticulum_test
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_test'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Endoplasmic_reticulum_test'
#     fasta2kmer(filename,filename_result)

#     # Extracellular_region_train
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_train'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Extracellular_region_train'
#     fasta2kmer(filename,filename_result)

#     # Extracellular_region_test
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_test'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Extracellular_region_test'
#     fasta2kmer(filename,filename_result)

#     # Mitochondria_train
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_train'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Mitochondria_train'
#     fasta2kmer(filename,filename_result)

#     # Mitochondria_test
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_test'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Mitochondria_test'
#     fasta2kmer(filename,filename_result)

#     # Nucleus_train
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_train'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Nucleus_train'
#     fasta2kmer(filename,filename_result)

#     # Nucleus_test
#     filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_test'
#     filename_result = '/home/zshen/workplace/new_mRNA/Data/kmer/k='+str(k)+'/Nucleus_test'
#     fasta2kmer(filename,filename_result)

# # Data2h5_k_mer

# # fasta2CGRSimage
# command = '/home/zshen/.conda/envs/pytorch/bin/python /home/zshen/workplace/new_mRNA/feature_extraction/dnacgr.py'
# def fastaCGRS(filename,filename_result):
#     path_list = glob.glob(filename+'/*')
#     # print(path_list[:10])

#     for path in path_list:
#         fin_command = command +' '+ path + ' --dest-dir '+ filename_result + ' --save  --dpi 50'
#         os.system(fin_command)
#         print(path.split('/')[-1])


# # 将fasta文件转化为CGR图片
# # Cytoplasm_train
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_train'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Cytoplasm_train'
# fastaCGRS(filename,filename_result)

# # Cytoplasm_test
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Cytoplasm_test'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Cytoplasm_test'
# fastaCGRS(filename,filename_result)

# # Endoplasmic_reticulum_train
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_train'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Endoplasmic_reticulum_train'
# fastaCGRS(filename,filename_result)

# # Endoplasmic_reticulum_test
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Endoplasmic_reticulum_test'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Endoplasmic_reticulum_test'
# fastaCGRS(filename,filename_result)

# # Extracellular_region_train
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_train'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Extracellular_region_train'
# fastaCGRS(filename,filename_result)

# # Extracellular_region_test
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Extracellular_region_test'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Extracellular_region_test'
# fastaCGRS(filename,filename_result)

# # Mitochondria_train
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_train'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Mitochondria_train'
# fastaCGRS(filename,filename_result)

# # Mitochondria_test
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Mitochondria_test'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Mitochondria_test'
# fastaCGRS(filename,filename_result)

# # Nucleus_train
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_train'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Nucleus_train'
# fastaCGRS(filename,filename_result)

# # Nucleus_test
# filename = '/home/zshen/workplace/new_mRNA/Data/fasta/Nucleus_test'
# filename_result = '/home/zshen/workplace/new_mRNA/Data/CGRS/Nucleus_test'
# fastaCGRS(filename,filename_result)

# # Data2h5_CGR.py

