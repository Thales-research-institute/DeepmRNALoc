from tensorflow import keras

def build_model(modelname,layer_size = 128,
                learning_rate = 1e-3,loss="sparse_categorical_crossentropy"):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+184*247]))

    if modelname == 'FCN':
        model.add(keras.layers.Dense(layer_size/2,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dense(layer_size,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dense(layer_size*2,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dense(layer_size*4,kernel_initializer='glorot_uniform'))

    model.add(keras.layers.Dense(5,activation="softmax"))
    model.compile(loss=loss,
                    optimizer = keras.optimizers.Adam(learning_rate,decay=1e-3 / 200),
                    metrics=['sparse_categorical_accuracy'])
    return model