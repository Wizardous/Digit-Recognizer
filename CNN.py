import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import os


class CNN(object):

    def __init__(self):
        # loading mnist data
        (X_train,y_train), (X_test,y_test) = mnist.load_data()

        # feature scaling and normalization
        self.training_images = X_train.reshape((60000, 28 , 28,1)).astype('float32') / 255
        self.training_targets = to_categorical(y_train)

        self.test_images = X_test.reshape((10000, 28 , 28,1)).astype('float32') / 255
        self.test_targets = to_categorical(y_test)

        self.input_shape = (self.training_images.shape[1],)

        # building the model
        self.model = Sequential()
        self.model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.training_images, self.training_targets, validation_split=0.3, callbacks=[EarlyStopping(patience=2)], epochs=25)
        self.save_model()

    def save_model(self):
        print("Saving model...")
        print('\nChecking saving dir...')
        if not os.path.isdir("./models"):
            print("model folder does not exists, creating one...")
            try:
                os.mkdir("./models")
            except OSError:
                print("Failed to create the folder, try running the code again after creating 'models' folder in same directory!")
            else:
                print("Folder created successfully!")
                self.model.save('models/mnistCNN.h5')
                print("Model Saved!")


if __name__ == "__main__":
    os.system("cls")
    print("Ready to build your CNN, hit return to start...")
    input()
    print("Building your Neural Net...")
    CNN_mod_1 = CNN()
