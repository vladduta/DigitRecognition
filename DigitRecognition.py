import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2

class MnistCNN(object):
    def __init__(self, epochs=10):
        self.batch_size = 128
        self.epochs = epochs
        self.num_classes = 10

        self.img_rows, self.img_cols = 28, 28
        self.input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)

        # Normalizare
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        # Conversia label-urilor in date binare categorice
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def save_model(self, name):
        if self.model is None:
            print("Model is not yet built")
            return
        self.model.save(name)

    def load_model(self, name):
        self.model = keras.models.load_model(name)

    def train(self):
        if self.model is None:
            print("Model is not yet built")
            return
        if self.x_train is None or self.y_train is None:
            print("Train data not loaded yet")
            return

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=1, validation_data=(self.x_test, self.y_test))

    def eval(self):
        if self.x_test is None or self.y_test is None:
            print("Test data not loaded yet")
            return

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Acuratete:', score[1] * 100.0, '%')

    def predict(self, input_img):
        if self.model is None:
            print("Model is not yet built")
            return

        if input_img is None:
            print("Input image not loaded properly")
            return None

        if len(input_img.shape) > 2:  # verificam daca imaginea de intrare are mai mult de 2 dimensiuni
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.resize(input_img, (self.img_rows, self.img_cols))

        # Normalizeaza imaginea de intrare
        input_img = input_img.astype('float32')
        input_img /= 255

        result = self.model.predict(np.reshape(input_img, (1, self.img_rows, self.img_cols, 1)))
        return np.argmax(result)

    def predict_on_image(self, path):
        # Predict
        img = cv2.imread(path)
        if img is None:
            print(f"Error: could not read image from path: {path}")
            return

        digit = self.predict(img)
        if digit is not None:
            # Show result
            print(f'Result: the image is a {digit}.')
            cv2.imshow('Input image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Prediction failed.")

if __name__ == '__main__':
    cnn = MnistCNN(epochs=10)
    cnn.load_data()
    cnn.build_model()

    cnn.train()
    # cnn.save_model('Model1')
    # cnn.load_model('Model1')
    cnn.predict_on_image(r"C:\Users\vladm\OneDrive\Desktop\SVA\nine.png")
