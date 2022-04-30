import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist


class CNN:
    def __init__(self):
        self.model = Sequential()
        self.modeltrained = False
        self.modelbuilt = False

    def build_model(self):
        if self.modelbuilt:
            return
        self.model.add(Conv2D(64,(5,5),input_shape=(28,28,1),activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Conv2D(64,(5,5),activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.modelbuilt = True

    def train_model(self):
        if not self.modelbuilt:
            raise Exception("Built model not found")
        if self.modeltrained:
            return
        mnist_data = mnist
        (x_train, y_train), (x_test, y_test) = mnist_data.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.model.fit(x=x_train, y=y_train, epochs=5)
        test_loss, test_acc = self.model.evaluate(x=x_test, y=y_test)
        print('CNN Test accuracy: {}'.format(test_acc))
        self.modeltrained = True

    def save_model(self):
        if not self.modelbuilt:
            raise Exception("Build and compile the model first!")
        if not self.modeltrained:
            raise Exception("Train and evaluate the model first!")
        self.model.save("cnn.hdf5", overwrite=True)
        print('CNN model saved as cnn.hdf5')
