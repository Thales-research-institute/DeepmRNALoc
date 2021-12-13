import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='[mRNALoc] XXXXX')

parser.add_argument('--model', type=str, required=False, default='mRNALoc',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, required=False, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, required=False, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--train_epochs', type=int, required=False, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, required=False, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, required=False, default='mse',help='loss function')

parser.add_argument('--use_gpu', type=bool, required=False, default=True, help='use gpu')
parser.add_argument('--extract_feature', type=bool, required=False, default=True, help='extract feature from data or not')



args = parser.parse_args()
print(args.model)
print(args.data)
print(args.root_path)
print(args.train_epochs)
print(args.batch_size)
print(args.patience)
print(args.learning_rate)
print(args.loss)
print(args.use_gpu)
print(args.extract_feature)

root_path = '/home/zshen/workplace/new_mRNA/git_tmp'


# preprocess data
# extract feature

# load train data(feature data and label)
from utils.load_date import load_data
x_train,x_test,y_train,y_test = load_data(root_path)


# init model
from models.FCN import build_model
model = build_model()

# init path
logdir = root_path
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"mRNA_model_indep.h5")

# train model
from utils.train import train_model
train_model(model,logdir,output_model_file,x_train,y_train)

# validation
from utils.evaluate import independent_test,classification_reports,confusion_mmatrix,calculate_accuracy
independent_test(model,output_model_file,x_train,x_test,y_train,y_test)

# classification_report
classification_reports(model,x_test,y_test)

# confusion matrix
confusion_mmatrix(model,x_test,y_test)

# calculate accuracy
calculate_accuracy(model,x_test,y_test)