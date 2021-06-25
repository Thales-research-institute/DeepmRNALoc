import numpy as np

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
# 将一个fasta文件切割为多个
#Cytoplasm_train
filename = './Data/data/Cytoplasm_train.fasta'
filename_result = './Data/fasta/Cytoplasm_train'
cut_fasta(filename,filename_result)

# Cytoplasm_test
filename = './Data/data/Cytoplasm_indep1.fasta'
filename_result = './Data/fasta/Cytoplasm_test'

cut_fasta(filename,filename_result)

# Endoplasmic_reticulum_train
filename = './Data/data/Endoplasmic_reticulum_train.fasta'
filename_result = './Data/fasta/Endoplasmic_reticulum_train'
cut_fasta(filename,filename_result)

# Endoplasmic_reticulum_test
filename = './Data/data/Endoplasmic_reticulum_indep1.fasta'
filename_result = './Data/fasta/Endoplasmic_reticulum_test'
cut_fasta(filename,filename_result)

# Extracellular_region_train
filename = './Data/data/Extracellular_region_train.fasta'
filename_result = './Data/fasta/Extracellular_region_train'
cut_fasta(filename,filename_result)

# Extracellular_region_test
filename = './Data/data/Extracellular_region_indep1.fasta'
filename_result = './Data/fasta/Extracellular_region_test'
cut_fasta(filename,filename_result)

# Mitochondria_train
filename = './Data/data/Mitochondria_train.fasta'
filename_result = './Data/fasta/Mitochondria_train'
cut_fasta(filename,filename_result)

# Mitochondria_test
filename = './Data/data/Mitochondria_indep1.fasta'
filename_result = './Data/fasta/Mitochondria_test'
cut_fasta(filename,filename_result)

# Nucleus_train
filename = './Data/data/Nucleus_train.fasta'
filename_result = './Data/fasta/Nucleus_train'
cut_fasta(filename,filename_result)

# Nucleus_test
filename = './Data/data/Nucleus_indep1.fasta'
filename_result = './Data/fasta/Nucleus_test'
cut_fasta(filename,filename_result)