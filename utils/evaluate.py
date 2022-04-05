import numpy as np
from sklearn.metrics import accuracy_score
# independent validation
def independent_test(model,path,x_train,x_test,y_train,y_test):
    model.load_weights(path)
    model.evaluate(x_test,y_test,batch_size=512)
    model.evaluate(x_train,y_train,batch_size=512)

# classification_report
from sklearn.metrics import classification_report
def classification_reports(model,x_test,y_test):
    y_pred = model.predict_classes(x_test)
    y_true = y_test
    labels = [0,1,2,3,4]
    target_names = ['Cytoplasm','Endoplasmic_reticulum','Extracellular_region','Mitochondria','Nucleus']
    print(classification_report(y_true,y_pred,labels=labels,target_names = target_names,digits=3))

# confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
def confusion_mmatrix(model,x_test,y_test):
    y_pred = model.predict_classes(x_test)
    y_true = y_test
    print(np.array(y_true).shape)
    print(np.array(y_pred).shape)
    print('confusion_matrix:'+'\n'+str(confusion_matrix(y_true,y_pred)))

# calculate accuracy
def calculate_accuracy(model,x_test,y_test):
    target_names = ['Cytoplasm','Endoplasmic_reticulum','Extracellular_region','Mitochondria','Nucleus']
    y_pred = model.predict_classes(x_test)
    y_true = y_test
    label_0 = 0
    label_true_0 = 0
    label_1 = 0
    label_true_1 = 0
    label_2 = 0
    label_true_2 = 0
    label_3 = 0
    label_true_3 = 0
    label_4 = 0
    label_true_4 = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            label_0 += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            label_true_0 += 1

        if y_true[i] == 1:
            label_1 += 1
        if y_true[i] == 1 and y_pred[i] == 1:
            label_true_1 += 1

        if y_true[i] == 2:
            label_2 += 1
        if y_true[i] == 2 and y_pred[i] == 2:
            label_true_2 += 1

        if y_true[i] == 3:
            label_3 += 1
        if y_true[i] == 3 and y_pred[i] == 3:
            label_true_3 += 1

        if y_true[i] == 4:
            label_4 += 1
        if y_true[i] == 4 and y_pred[i] == 4:
            label_true_4 += 1

    print(target_names[0]+':'+str(label_true_0/label_0))
    print(target_names[1]+':'+str(label_true_1/label_1))
    print(target_names[2]+':'+str(label_true_2/label_2))
    print(target_names[3]+':'+str(label_true_3/label_3))
    print(target_names[4]+':'+str(label_true_4/label_4))
    print('accuracy_all: '+str(accuracy_score(y_true,y_pred)))