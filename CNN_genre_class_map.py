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

    # Layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
        # Batch Normalization normalizes/standardizes the activations in a current layer
            # Speed up the model and make it more reliable
    model.add(keras.layers.BatchNormalization())

    # Layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Flatten output and feed it into dense layer; from 2D to 1D
    model.add(keras.layers.Flatten())
        # Creates a fully connected layer
            # Args. number of neurons and the type of activation
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Output layer
        # Number of output neurons is the number of labels you want to predict
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y, genre_mapping):

    X = X[np.newaxis, ...]

    # X is a 3D array; Predict needs a 4D array
        # The prediction will be a 2D array, containing the scores for each genre
    prediction = model.predict(X)

    # Extract index with max value
        # Gets a 1D array; the index that's predicted
    predicted_index = np.argmax(prediction, axis=1)

    print('Expected index: {}, Predicted index: {}'.format(y, predicted_index))
    print('Predicted Genre: {}'.format(genre_mapping[predicted_index]))

if __name__=='__main__':
    
    # Create train, validation, and test sets
        # Prepare datasets args:
            # size of the training set
            # size of the validation set; taken from the training set
    x_train, x_validation, x_test, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)


    # Build the CNN network
        # Creates 3D shapes out of 4D x_train array
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)


    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train the CNN
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                                                batch_size=32,
                                                epochs=50)


    # Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Accuracy on test set is: {}'.format(test_accuracy))

    # # Make predictions on a sample
    # X = x_test[100]
    # y = y_test[100]

    # predict(model, X, y, genre_mapping)