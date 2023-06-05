import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

data_path = 'data.json'

def load_data(data_path):

    global genre_mapping

    with open(data_path, 'r') as fp:
        data = json.load(fp)
    x = np.array(data['mfcc'])
    y = np.array(data['labels'])
    genre_mapping = np.array(data['mapping'])
    return x, y

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
    
    # Tensforflow needs a 3D array for each sample
    # mfcc coefficients, no. of samples/hop length, depth/channel
    x_train = x_train[..., np.newaxis]
        # Creates a 4D array -> (total no. of samples, no. of samples, no. of sample / hop length, depth)    
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, x_test, y_train, y_validation, y_test

def build_model(input_shape):

    # Create model
    model = keras.Sequential()

    # Convolutional Layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(3, strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Convolutional Layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(3, strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Convolutional Layer 3
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(3, strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Convolutional Layer 4
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(3, strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Convolutional Layer 5
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(3, strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Recurrent Layer
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.LSTM(64, return_sequences=True))
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
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
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
    
    # Evaluate Model on Test Set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest Accuracy: {}'.format(test_acc))