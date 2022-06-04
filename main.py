import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='[DeepmRNALoc] A novel predictor of eukaryotic mRNA subcellular localization based on deep learning')

parser.add_argument('--model', type=str, required=True, default='DeepmRNALoc',help='model of experiment, options: [DeepmRNALoc,FCN]')
# parser.add_argument('--root_path', type=str, required=False, default='', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, required=False, default='checkpoints/', help='The path of model checkpoints')
parser.add_argument('--train', action='store_true', help='whether the model will be trained')
parser.add_argument('--train_epochs', type=int, required=False, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, required=False, default=None, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, required=False, default=1e-3, help='optimizer learning rate')


args = parser.parse_args()
print(args.model)
# print(args.root_path)
print(args.train_epochs)
print(args.batch_size)
print(args.patience)
print(args.learning_rate)

root_path = '/home/zshen/Workplace/workplace/DeepmRNALoc_test'


# preprocess data
# extract feature

# load train data(feature data and label)
from utils.load_data import load_data
x_train,x_test,y_train,y_test = load_data(root_path)

# init model
from models.models import build_model
model = build_model(modelname=args.model,learning_rate=args.learning_rate)

# init path
ckpdir = root_path + os.sep + args.checkpoints
if not os.path.exists(ckpdir):
    os.mkdir(ckpdir)
output_model_file = os.path.join(ckpdir,"mRNA_model_indep.h5")

# train model
from utils.train import train_model
if args.train == True:
    train_model(model,output_model_file,x_train,y_train,args.train_epochs,args.batch_size,args.patience)

# validation
from utils.evaluate import independent_test,classification_reports,confusion_mmatrix,calculate_accuracy
independent_test(model,output_model_file,x_train,x_test,y_train,y_test,args.batch_size)

# classification_report
classification_reports(model,x_test,y_test)

# confusion matrix
confusion_mmatrix(model,x_test,y_test)

# calculate accuracy
calculate_accuracy(model,x_test,y_test)