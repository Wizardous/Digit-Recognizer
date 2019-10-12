# Importing the Keras libraries and packages
from keras.models import load_model
import os
import numpy as np
import cv2
from CNN import CNN

checkdir = 'models/mnistCNN.h5'
if os.path.isfile(checkdir):
    print("CNN model Exists!")
else:
    print('Model does not exists, need create one (might take a while!)')
    print("Hit return to start the Traning...")
    input()
    print("\n\n\n")
    cnn = CNN()


model = load_model(checkdir)


class NeuralNet(object):

    def predict(self, Image):
        input = cv2.resize(Image, (28 , 28)).reshape((28 , 28,1)).astype('float32') / 255
        # Predicting the Test set results
        y_pred = model.predict_classes(np.array([input]))
        os.system("cls")
        print("Digit Predicted as: {}".format(y_pred[0]))
        return y_pred