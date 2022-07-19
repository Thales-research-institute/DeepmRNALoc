from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional

def build_model(modelname,layer_size = 128,
                learning_rate = 1e-3,
                dropout_rate = 0.3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+184*247]))
    if modelname == "DeepmRNALoc":
        model.add(keras.layers.Reshape((4**1+4**2+4**3+4**4+4**5+4**6+4**7+4**8+184*247,1)))
        model.add(keras.layers.Conv1D(64, 3,strides=2,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Conv1D(64, 3,strides=1,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Conv1D(128, 3,strides=2,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Conv1D(128, 3,strides=1,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Conv1D(256, 3,strides=2,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Conv1D(256, 3,strides=1,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Conv1D(512, 3,strides=2,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Conv1D(512, 3,strides=1,padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(dropout_rate))
    #LSTM
        model.add(Bidirectional(keras.layers.CuDNNLSTM(512, return_sequences=True)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(Bidirectional(keras.layers.CuDNNLSTM(512, return_sequences=False)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dropout(dropout_rate))
    #FCN
        model.add(keras.layers.Dense(layer_size,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dense(layer_size*2,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(keras.layers.Dense(layer_size*4,kernel_initializer='glorot_uniform'))
        model.add(keras.layers.Dense(5,activation="softmax"))
        loss = CategoricalCrossentropy(label_smoothing=0.01)
        model.compile(loss=loss,
                        optimizer = keras.optimizers.Adam(learning_rate,decay=1e-3 / 200),
                        metrics=['categorical_accuracy'])
    return model
