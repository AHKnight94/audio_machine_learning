import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'data.json'

def load_data(data_path):


    with open(data_path, 'r') as fp:
        data = json.load(fp)
    x = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return x, y

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(test_size, validation_size):

    # Load data
    x, y = load_data(data_path)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # Validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train,
                                                                    y_train,
                                                                    test_size=validation_size
                                                                    )

    return x_train, x_validation, x_test, x_test, y_train, y_validation, y_test

def build_model(input_shape):

    # Create model
    model = keras.Sequential()

    # Build 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # Dense Layer
        # return_sequences defaults to false
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Output layer
        # Number of output neurons is the number of labels you want to predict
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__=='__main__':
        
    # Create train, validation, and test sets
    # Prepare datasets args:
        # size of the training set
        # size of the validation set; taken from the training set
    x_train, x_validation, x_test, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Build the LSTM network
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train Model
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_validation, y_validation),
                        batch_size=32,
                        epochs=50)
    
    # # Plot Accuracy/Error for Training and Validation
    # plot_history(history)

    # Evaluate Model on Test Set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest Accuracy: {}'.format(test_acc))