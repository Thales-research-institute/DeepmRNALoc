import imp
from tensorflow import keras
import numpy as np
def get_one_hot(arr,num_classes):
    res = np.eye(num_classes)[arr]
    return res

# train and store model
def train_model(model,output_model_file,x_train,y_train,epoch_num=200,batch_size=32,patience_num=None):
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_model_file,
                                        save_best_only = True)
    ]
    if patience_num != None:
        callbacks.append(keras.callbacks.EarlyStopping(patience=patience_num,min_delta=1e-3))
    from sklearn.model_selection import KFold
    KF = KFold(n_splits = 5,random_state = 7,shuffle=True)
    for train_index, valid_index in KF.split(x_train):
        x_train2 = x_train[train_index]
        y_train2 = y_train[train_index]
        x_valid2 = x_train[valid_index]
        y_valid2 = y_train[valid_index]

        y_train2 = get_one_hot(y_train2,5)
        y_valid2 = get_one_hot(y_valid2,5)
        print("train_batch_size:",batch_size)
        model.fit(x_train2,y_train2,epochs=epoch_num,batch_size=batch_size
                        ,validation_data=(x_valid2,y_valid2),callbacks=callbacks)
        print(model.summary())