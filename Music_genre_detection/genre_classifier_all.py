import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import copy

DATA_PATH = "data.json"

def load_data(data_path):
    """Loads training dataset from json file

        :param data_path (str): Path to json file containing data info
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(test_size, validation_size):
    """Splitting the train set into test set and validation set sequentially

        :param test_size (float): proportion of the dataset to include in the test split
        :param validation_size (float): proportion of the dataset to include in the validation split

    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_nn_model(input_shape):
    """Generate NN model

        :param: input_shape (tuple): shape of the input
        :return model: NN model
    """    
    model = keras.Sequential([
        # input layer
        # multi demensional array and flatten it out
        # inputs.shape[1]: the intervals
        # inputs.shape[2]: the value of the mfcc for that intervals
        keras.layers.Flatten(input_shape=input_shape),
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        # softmax: the sum of the result of all the labels = 1
        # predicting: pick the neuron hav highest value
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def build_cnn_model(input_shape):
    """Generate CNN model

        :param: input_shape (tuple): shape of the input
        :return model: CNN model
    """    
    # create model
    model = keras.Sequential()

    # 1st convolution layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same')) # padding='same' is zero-padding
    model.add(keras.layers.BatchNormalization()) # standadized the activation layer -> speed up training process

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def build_lstm_model(input_shape):
    """Generate RNN-LSTM model

        :param: input_shape (tuple): shape of the input
        :return model: RNN-LSTM model
    """
    
    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    # return_sequences = TRUE => sequence to sequence RNN layer
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True)) 
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model    


if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # lstm and normal feed forward nn can use the same 
    # train validate and test set input

    # train, test and validation set for cnn
    X_train_cnn = copy.deepcopy(X_train)
    X_train_cnn = X_train_cnn[..., np.newaxis]
    X_validation_cnn = copy.deepcopy(X_validation)
    X_validation_cnn = X_validation_cnn[..., np.newaxis]
    X_test_cnn = copy.deepcopy(X_test)
    X_test_cnn = X_test_cnn[..., np.newaxis]

    # all 3 models used the same optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    # NN model
    print("[NN] Model compiling: ")
    input_shape_nn = X_train[0].shape
    nn_model = build_nn_model(input_shape_nn)
    nn_model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    history_nn = nn_model.fit(X_train, y_train, 
                            validation_data=(X_validation,y_validation),
                            epochs=30,
                            batch_size=32)
    plot_history(history_nn)

    # CNN model
    print("[CNN] Model compiling: ")
    input_shape_cnn = X_train_cnn[0].shape
    cnn_model = build_cnn_model(input_shape_cnn)
    cnn_model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    history_cnn = cnn_model.fit(X_train_cnn, y_train, 
                                validation_data=(X_validation_cnn,y_validation),
                                epochs=30,
                                batch_size=32)
    plot_history(history_cnn)

    # LSTM model
    print("[LSTM] Model compiling: ")
    input_shape_lstm = X_train[0].shape
    lstm_model = build_lstm_model(input_shape_lstm)
    lstm_model.compile(optimizer=optimizer,
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
    history_lstm = lstm_model.fit(X_train, y_train, 
                                  validation_data=(X_validation,y_validation),
                                  epochs=30,
                                  batch_size=32)
    plot_history(history_lstm)


    # test acc on 3 models
    test_error_nn, test_acc_nn = nn_model.evaluate(X_test, y_test, verbose=2)
    print(f"[NN] Test accuracy: {test_acc_nn}")

    test_error_cnn, test_acc_cnn = cnn_model.evaluate(X_test_cnn, y_test, verbose=2)
    print(f"[CNN] Test accuracy: {test_acc_cnn}")

    test_error_lstm, test_acc_lstm = lstm_model.evaluate(X_test, y_test, verbose=2)
    print(f"[LSTM] Test accuracy: {test_acc_lstm}")