# create folder
import os
import numpy as np
import pandas as pd
import glob
import cv2

prefix = './data'

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
        path2 = path + '/kmer'
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

    path5 = path + '/H5'
    mkdir(path5)





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
    filename.append(prefix + '/Cytoplasm_train.fasta')
    filename_result.append('./Data/fasta/Cytoplasm_train')

    # Cytoplasm_test
    filename.append(prefix + '/Cytoplasm_indep1.fasta')
    filename_result.append('./Data/fasta/Cytoplasm_test')

    # Endoplasmic_reticulum_train
    filename.append(prefix + '/Endoplasmic_reticulum_train.fasta')
    filename_result.append('./Data/fasta/Endoplasmic_reticulum_train')

    # Endoplasmic_reticulum_test
    filename.append(prefix + '/Endoplasmic_reticulum_indep1.fasta')
    filename_result.append('./Data/fasta/Endoplasmic_reticulum_test')


    # Extracellular_region_train
    filename.append(prefix + '/Extracellular_region_train.fasta')
    filename_result.append('./Data/fasta/Extracellular_region_train')

    # Extracellular_region_test
    filename.append(prefix + '/Extracellular_region_indep1.fasta')
    filename_result.append('./Data/fasta/Extracellular_region_test')


    # Mitochondria_train
    filename.append(prefix + '/Mitochondria_train.fasta')
    filename_result.append('./Data/fasta/Mitochondria_train')

    # Mitochondria_test
    filename.append(prefix + '/Mitochondria_indep1.fasta')
    filename_result.append('./Data/fasta/Mitochondria_test')

    # ./Data
    # Nucleus_train
    filename.append(prefix + '/Nucleus_train.fasta')
    filename_result.append('./Data/fasta/Nucleus_train')

    # Nucleus_test
    filename.append(prefix + '/Nucleus_indep1.fasta')
    filename_result.append('./Data/fasta/Nucleus_test')



    for i in range(len(filename)):
        print(filename[i])
        print(filename_result[i])
        cut_fasta(filename[i],filename_result[i])


# kmer
def get_tris(k):
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars)**k
    for i in range(0, end):
        n = i
        add = ''
        for j in range(k):
            ch = chars[n % base]
            n = int(n/base)
            add += ch
        nucle_com.append(add)
    return nucle_com

def get_kmer(path,k):
    fasta = open(path)
    fasta = fasta.read()
    sequence = "".join(fasta.split("\n")[1:])
    sequence = sequence.replace("N", "")
    print(len(sequence))
    kmerbases = get_tris(k)

    kmermap = {}
    for kmer in  kmerbases:
        kmermap[kmer] = 0

    # print(kmermap)
    for index in range(len(sequence)-k+1):
        kmermap[sequence[index:index+k]] += 1

    # print(kmermap)
    # print(len(kmermap))
    result = []
    for kmer in kmermap:
        result.append(kmermap[kmer])
    return result


def fasta2kmer(filename,filename_result,k):
    path_list = glob.glob(filename+'/*')
    # print(path_list[:10])
    column_names = [0]
    for path in path_list:
        seq = get_kmer(path,k)
        # print(seq)
        names = path.split('/')
        print(names[-1])
        name = names[-1][:-6]
        # print(name)
        test=pd.DataFrame(columns=column_names,data=seq)
        test.to_csv(filename_result+'/'+name+'.txt',encoding='gbk')

def extract_kmer():
    start_k = 1
    end_k = 8
    for k in range(start_k,end_k+1):
        # 将fasta文件转化问kmer
        #Cytoplasm_train
        filename = './Data/fasta/Cytoplasm_train'
        filename_result = './Data/kmer/k='+str(k)+'/Cytoplasm_train'
        fasta2kmer(filename,filename_result,k)

        # Cytoplasm_test
        filename = './Data/fasta/Cytoplasm_test'
        filename_result = './Data/kmer/k='+str(k)+'/Cytoplasm_test'
        fasta2kmer(filename,filename_result,k)

        # Endoplasmic_reticulum_train
        filename = './Data/fasta/Endoplasmic_reticulum_train'
        filename_result = './Data/kmer/k='+str(k)+'/Endoplasmic_reticulum_train'
        fasta2kmer(filename,filename_result,k)

        # Endoplasmic_reticulum_test
        filename = './Data/fasta/Endoplasmic_reticulum_test'
        filename_result = './Data/kmer/k='+str(k)+'/Endoplasmic_reticulum_test'
        fasta2kmer(filename,filename_result,k)

        # Extracellular_region_train
        filename = './Data/fasta/Extracellular_region_train'
        filename_result = './Data/kmer/k='+str(k)+'/Extracellular_region_train'
        fasta2kmer(filename,filename_result,k)

        # Extracellular_region_test
        filename = './Data/fasta/Extracellular_region_test'
        filename_result = './Data/kmer/k='+str(k)+'/Extracellular_region_test'
        fasta2kmer(filename,filename_result,k)

        # Mitochondria_train
        filename = './Data/fasta/Mitochondria_train'
        filename_result = './Data/kmer/k='+str(k)+'/Mitochondria_train'
        fasta2kmer(filename,filename_result,k)

        # Mitochondria_test
        filename = './Data/fasta/Mitochondria_test'
        filename_result = './Data/kmer/k='+str(k)+'/Mitochondria_test'
        fasta2kmer(filename,filename_result,k)

        # Nucleus_train
        filename = './Data/fasta/Nucleus_train'
        filename_result = './Data/kmer/k='+str(k)+'/Nucleus_train'
        fasta2kmer(filename,filename_result,k)

        # Nucleus_test
        filename = './Data/fasta/Nucleus_test'
        filename_result = './Data/kmer/k='+str(k)+'/Nucleus_test'
        fasta2kmer(filename,filename_result,k)

# Data2h5_kmer
def Data2h5_kmer():
    classnamemap = {'Cytoplasm':0,'Endoplasmic_reticulum':1,'Extracellular_region':2,'Mitochondria':3,'Nucleus':4}

    x_train_all_k1 = []
    y_train_all_k1 = []
    x_test_k1 = []
    y_test_k1 = []

    path_kmer = './Data/kmer/k=1'

    def get_key(text):
        t = text.split('/')[-1][8:-4]
        return int(t)

    def get_key_last(text):
        t = text.split('/')[-1][:-4]
        return t

    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
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

    path_kmer = './Data/kmer/k=2'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
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

    path_kmer = './Data/kmer/k=3'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
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

    path_kmer = './Data/kmer/k=4'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
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

    path_kmer = './Data/kmer/k=5'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
            csv = pd.read_csv(x2)
            # print(csv)
            # print(csv.shape)
            if trainortest == 'train':
                x_train_all_k5.append(csv.values[:,1:])
                y_train_all_k5.append(classnamemap[classname])
            elif trainortest == 'test':
                x_test_k5.append(csv.values[:,1:])
                y_test_k5.append(classnamemap[classname])


    x_train_all_k6 = []
    y_train_all_k6 = []
    x_test_k6 = []
    y_test_k6 = []

    path_kmer = './Data/kmer/k=6'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
            csv = pd.read_csv(x2)
            # print(csv)
            # print(csv.shape)
            if trainortest == 'train':
                x_train_all_k6.append(csv.values[:,1:])
                y_train_all_k6.append(classnamemap[classname])
            elif trainortest == 'test':
                x_test_k6.append(csv.values[:,1:])
                y_test_k6.append(classnamemap[classname])



    x_train_all_k7 = []
    y_train_all_k7 = []
    x_test_k7 = []
    y_test_k7 = []

    path_kmer = './Data/kmer/k=7'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
            csv = pd.read_csv(x2)
            # print(csv)
            # print(csv.shape)
            if trainortest == 'train':
                x_train_all_k7.append(csv.values[:,1:])
                y_train_all_k7.append(classnamemap[classname])
            elif trainortest == 'test':
                x_test_k7.append(csv.values[:,1:])
                y_test_k7.append(classnamemap[classname])

    x_train_all_k8 = []
    y_train_all_k8 = []
    x_test_k8 = []
    y_test_k8 = []

    path_kmer = './Data/kmer/k=8'


    path_list = glob.glob(path_kmer+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
            csv = pd.read_csv(x2)
            # print(csv)
            # print(csv.shape)
            if trainortest == 'train':
                x_train_all_k8.append(csv.values[:,1:])
                y_train_all_k8.append(classnamemap[classname])
            elif trainortest == 'test':
                x_test_k8.append(csv.values[:,1:])
                y_test_k8.append(classnamemap[classname])


    import numpy as np
    x_train_all_k1 = np.array(x_train_all_k1)
    x_train_all_k2 = np.array(x_train_all_k2)
    x_train_all_k3 = np.array(x_train_all_k3)
    x_train_all_k4 = np.array(x_train_all_k4)
    x_train_all_k5 = np.array(x_train_all_k5)
    x_train_all_k6 = np.array(x_train_all_k6)
    x_train_all_k7 = np.array(x_train_all_k7)
    x_train_all_k8 = np.array(x_train_all_k8)

    print(x_train_all_k1.shape)
    print(x_train_all_k2.shape)
    print(x_train_all_k3.shape)
    print(x_train_all_k4.shape)
    print(x_train_all_k5.shape)
    print(x_train_all_k6.shape)
    print(x_train_all_k7.shape)
    print(x_train_all_k8.shape)

    x_train_all = np.concatenate((x_train_all_k1,x_train_all_k2,x_train_all_k3,x_train_all_k4,x_train_all_k5,x_train_all_k6,x_train_all_k7,x_train_all_k8),axis=1)
    print(x_train_all.shape)


    x_test_all_k1 = np.array(x_test_k1)
    x_test_all_k2 = np.array(x_test_k2)
    x_test_all_k3 = np.array(x_test_k3)
    x_test_all_k4 = np.array(x_test_k4)
    x_test_all_k5 = np.array(x_test_k5)
    x_test_all_k6 = np.array(x_test_k6)
    x_test_all_k7 = np.array(x_test_k7)
    x_test_all_k8 = np.array(x_test_k8)

    print(x_test_all_k1.shape)
    print(x_test_all_k2.shape)
    print(x_test_all_k3.shape)
    print(x_test_all_k4.shape)
    print(x_test_all_k5.shape)
    print(x_test_all_k6.shape)
    print(x_test_all_k7.shape)
    print(x_test_all_k8.shape)

    x_test_all = np.concatenate((x_test_all_k1,x_test_all_k2,x_test_all_k3,x_test_all_k4,x_test_all_k5,x_test_all_k6,x_test_all_k7,x_test_all_k8),axis=1)
    print(x_test_all.shape)


    y_train_all = y_train_all_k1
    y_test_all = y_test_k1

    
    import h5py
    import numpy as np

    with h5py.File("./Data/H5/mRNA_kmer_k=12345678.h5", 'w') as f:
        f.create_dataset('mRNA_included',data=np.array(['Cytoplasm'.encode(),'Endoplasmic_reticulum'.encode(),'Extracellular_region'.encode(),'Mitochondria'.encode(),'Nucleus'.encode()]))
        f.create_dataset('kmer_x_train_all',data = x_train_all)
        f.create_dataset('kmer_y_train_all',data = y_train_all)
        f.create_dataset('kmer_x_test',data = x_test_all)
        f.create_dataset('kmer_y_test',data = y_test_all)

# fasta2CGRSimage
def fastaCGRS(filename,filename_result):
    # need to change python virtual environment path !!!!
    command = '/home/zhehan/anaconda3/envs/mRNA/bin/python ./utils/dnacgr.py'
    path_list = glob.glob(filename+'/*')
    # print(path_list[:10])

    for path in path_list:
        fin_command = command +' '+ path + ' --dest-dir '+ filename_result + ' --save  --dpi 50'
        os.system(fin_command)
        print(path.split('/')[-1])


def use_CGRS_fasta2img():
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

# Data2h5_CGR.py
def get_key(text):
    t = text.split('/')[-1][8:-4]
    return int(t)

def get_key_last(text):
    t = text.split('/')[-1][:-4]
    return t

def Data2h5_CGR():
    classnamemap = {'Cytoplasm':0,'Endoplasmic_reticulum':1,'Extracellular_region':2,'Mitochondria':3,'Nucleus':4}
    # 先获取数据
    x_train_all = []
    y_train_all = []
    x_test = []
    y_test = []

    path_CGRS = './Data/CGRS'
    path_list = glob.glob(path_CGRS+'/*')
    path_list = sorted(path_list , key=get_key_last)
    for x in path_list:
        # print(x)
        folder_name = x.split('/')[-1]
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
        path_list_next = glob.glob(x+'/*')
        path_list_next = sorted(path_list_next , key=get_key)
        for x2 in path_list_next:
            img = cv2.imread(x2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[30:214, 41:288]
            # print(csv)
            # print(csv.shape)
            if trainortest == 'train':
                x_train_all.append(img)
                y_train_all.append(classnamemap[classname])
            elif trainortest == 'test':
                x_test.append(img)
                y_test.append(classnamemap[classname])


    import numpy as np
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
    with h5py.File("./Data/H5/mRNA_CGRS_cut.h5", 'w') as f:
        f.create_dataset('mRNA_included',data=np.array(['Cytoplasm'.encode(),'Endoplasmic_reticulum'.encode(),'Extracellular_region'.encode(),'Mitochondria'.encode(),'Nucleus'.encode()]))
        f.create_dataset('CGR_x_train_all',data = x_train_all)
        f.create_dataset('CGR_y_train_all',data = y_train_all)
        f.create_dataset('CGR_x_test',data = x_test)
        f.create_dataset('CGR_y_test',data = y_test)

def extract_feature():
    create_folder()
    cut_fastas()
    extract_kmer()
    Data2h5_kmer()
    use_CGRS_fasta2img()
    Data2h5_CGR()

extract_feature()