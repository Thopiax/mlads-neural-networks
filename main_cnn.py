import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical, normalize
from keras import backend as K
from data import load_data


def main():
    batch_size = 128
    epochs = 20

    training_data, test_data, validation_data = load_data("data4students.mat")

    print(training_data.data[0])
    print(training_data.targets[0])

    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(900,)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(7, activation='softmax'))


    assert K.image_data_format() == "channels_last"

    training_data.data = training_data.data.reshape(training_data.data.shape[0], 30, 30, 1)
    validation_data.data = validation_data.data.reshape(validation_data.data.shape[0], 30, 30, 1)
    test_data.data = test_data.data.reshape(test_data.data.shape[0], 30, 30, 1)
    input_shape = (30, 30, 1)

    print(training_data.data[0])


    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(training_data.data, training_data.targets,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(validation_data.data, validation_data.targets))

    score = model.evaluate(test_data.data, test_data.targets, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    main()
