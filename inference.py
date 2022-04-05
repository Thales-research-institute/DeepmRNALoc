import numpy as np
import pandas as pd

y_pred_all = []

command = '/home/zhehan/anaconda3/envs/mRNA/bin/python /home/zhehan/Workplace/mRNALoc/utils/dnacgr_forweb.py'
def fasta2CGRS(filename,filename_result):
    print('---- Create CGR figure start! ----')
    fin_command = command +' '+ filename + ' --dest-dir '+ filename_result + ' --name '+ ' tmp ' +' --save  --dpi 50 '
    os.system(fin_command)
    print(filename.split('/')[-1])
    print('---- Create CGR figure end! ----\n\n')

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

    for index in range(len(sequence)-k+1):
        kmermap[sequence[index:index+k]] += 1
    result = []
    for kmer in kmermap:
        result.append(kmermap[kmer])
    return result


# 构建模型 载入模型的参数
# model define
import os
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier


def build_model(layer_size = 128,
                learning_rate = 1e-3):
    model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+230*300]))
    model.add(keras.layers.Flatten(input_shape=[4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+184*247]))
    # model.add(keras.layers.Flatten(input_shape=[4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+240*320]))

    model.add(keras.layers.Dense(layer_size/2,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size*2,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size*4,kernel_initializer='glorot_uniform'))

    model.add(keras.layers.Dense(5,activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer = keras.optimizers.Adam(learning_rate,decay=1e-3 / 200),
                    metrics=['sparse_categorical_accuracy'])
    return model



model = build_model()

print('---- Init model start! ----')
logdir = '/home/zhehan/Workplace/mRNALoc/checkpoints/Web'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"mRNA_model_indep.h5")
print(output_model_file)
model.load_weights(output_model_file)
print('---- Init model end! ----')



# filename = '/home/zhehan/Workplace/mRNALoc/Data/data/Nucleus_indep1.fasta'
# filename = '/home/zhehan/Workplace/mRNALoc/iLoc/Nucleus.fasta'
# filename = '/home/zhehan/Workplace/mRNALoc/web_store/savedfile.fasta'
# filename = '/home/zhehan/Workplace/mRNALoc/iLoc/Endoplasmic_reticulum.fasta'
# filename = '/home/zhehan/Workplace/mRNALoc/iLoc/Cytosol.fasta'
filename = '/home/zhehan/Workplace/mRNALoc/web_store/savedfile1.txt'
# 提取fasta中的序列
seq = []
name = []
n = 0
with open(filename) as fs:
    for line in fs:
        if n % 2 == 0:
            name.append(line)
        else:
            seq.append(line)
        n += 1

# 处理序列后的换行符
for i in range(len(seq)):
    while seq[i][-1] == '\n':
        seq[i] = seq[i][:-1]
print(seq[0][-1])
print(seq[0])


# get feature CGR
# 预设一个文件夹 用于存放fasta文件和图片 之后进行读取
import os
import shutil
import glob
print('---- Store fasta start! ----')
# 1、存储fasta文件
if os.path.exists("/home/zhehan/Workplace/mRNALoc/tmp/fasta/"):
    shutil.rmtree("/home/zhehan/Workplace/mRNALoc/tmp/fasta/")
# print(os.path)
os.mkdir("/home/zhehan/Workplace/mRNALoc/tmp/fasta/")
for i in range(len(seq)):
    with open("/home/zhehan/Workplace/mRNALoc/tmp/fasta/{}.fasta".format(i),'w') as fs:
        fs.writelines(name[i])
        fs.writelines(seq[i])
    print(name[i])
print('---- Store fasta end! ----')

name = []
fasta_list = glob.glob('/home/zhehan/Workplace/mRNALoc/tmp/fasta/*')
for fasta_path in fasta_list:
    # 提取CGR图
    filename = fasta_path
    filename_result = '/home/zhehan/Workplace/mRNALoc/tmp/CGR'
    fasta2CGRS(filename,filename_result)
    # get name
    with open(filename) as fs:
        n = 0
        for line in fs:
            if n % 2 == 0:
                name.append('name' + str(line))
                print(line)
            n += 1
########### ？？？？？？？？？？？？？？？？？？？？所有CGR在kmer前就跑完了

    # get feature k_mer = 1 2 3 4 5 6 7 8
    k_mer1 = []
    k_mer2 = []
    k_mer3 = []
    k_mer4 = []
    k_mer5 = []
    k_mer6 = []
    k_mer7 = []
    k_mer8 = []

    print('---- Extracte kmer feature start! ----')

    k_mer1.append(get_kmer(fasta_path, 1))
    print(np.array(k_mer1).shape)

    k_mer2.append(get_kmer(fasta_path, 2))
    print(np.array(k_mer2).shape)

    k_mer3.append(get_kmer(fasta_path, 3))
    print(np.array(k_mer3).shape)

    k_mer4.append(get_kmer(fasta_path, 4))
    print(np.array(k_mer4).shape)

    k_mer5.append(get_kmer(fasta_path, 5))
    print(np.array(k_mer5).shape)

    k_mer6.append(get_kmer(fasta_path, 6))
    print(np.array(k_mer6).shape)

    k_mer7.append(get_kmer(fasta_path, 7))
    print(np.array(k_mer7).shape)

    k_mer8.append(get_kmer(fasta_path, 8))
    print(np.array(k_mer8).shape)


    k_mer = np.concatenate((k_mer1,k_mer2,k_mer3,k_mer4,k_mer5,k_mer6,k_mer7,k_mer8),axis = 1)

    print(np.array(k_mer).shape)

    print('---- Extracte kmer feature end! ----')


    # 获取CGR图片的特征
    import cv2

    CGR = []

    path_CGR_figure = '/home/zhehan/Workplace/mRNALoc/tmp/CGR/tmp.png'


    print('---- Extracte CGR feature start! ----')
    img = cv2.imread(path_CGR_figure)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[30:214, 41:288]
    # img = img[10:240, 10:310]
    # img = img[0:240, 0:320]
    CGR.append(img)

    CGR = np.array(CGR)
    # CGR = CGR.reshape(-1,240*320)
    CGR = CGR.reshape(-1,184*247)
    CGR = CGR/255.0
    print(CGR.shape)
    print('---- Extracte kmer feature end! ----')

    # 合并特征
    # k_mer = np.squeeze(k_mer)
    print(np.array(k_mer).shape)
    print(np.array(CGR).shape)
    test_x = np.concatenate((k_mer,CGR),axis = 1)

    print(np.array(test_x).shape)

    # 标准化
    import pickle
    f = open('/home/zhehan/Workplace/mRNALoc/checkpoints/Web/shuffle_scalar.pkl','rb')
    scaler = pickle.load(f)
    test_x = scaler.transform(test_x)


    # 预测
    target_names = ['Cytoplasm','Endoplasmic_reticulum','Extracellular_region','Mitochondria','Nucleus']
    y_pred = model.predict_classes(test_x)
    y_pred_all.extend(y_pred)
    print("pred: "+ target_names[y_pred[0]])
    print('----------------------------\n\n')

# 保存结果
res = 0
with open("/home/zhehan/Workplace/mRNALoc/web_store/savedfile2.txt",'w') as fs:
    for n in range(len(y_pred_all)):
        fs.writelines("{} : {}".format(name[n][:-1],target_names[y_pred_all[n]]+'\n'))
        if y_pred_all[n] == 4:
            res += 1
print("{} / {}".format(res,len(y_pred_all)))
print(res/len(y_pred_all))
