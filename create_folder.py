import os 

def mkdir(path):
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
 
	else:
		print(path+"folder already exists!")



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



for i in range(1,6):
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