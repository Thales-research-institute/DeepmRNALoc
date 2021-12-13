from tensorflow import keras
# train and store model
def train_model(model,logdir,output_model_file,x_train,y_train):
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file,
                                        save_best_only = True),
        # keras.callbacks.EarlyStopping(patience=30,min_delta=1e-3)
    ]
    from sklearn.model_selection import KFold
    KF = KFold(n_splits = 5,random_state = 7,shuffle=True)
    for train_index, valid_index in KF.split(x_train):
        x_train2 = x_train[train_index]
        y_train2 = y_train[train_index]
        x_valid2 = x_train[valid_index]
        y_valid2 = y_train[valid_index]
        model.fit(x_train2,y_train2,epochs=8,batch_size=4
                        ,validation_data=(x_valid2,y_valid2),callbacks=callbacks)