from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.regularizers import l2

import matplotlib.pyplot as plt

class Model_Class:
    def __init__(self, settings):
        self.inputShape = settings["input_shape"]
        self.kernalRegularizer = l2(settings["l2_weight"])
        self.activationMethod = settings["activation"]
        self.lossFunction = settings["loss"]
        self.optimizerMethod = settings["optimizer"]
        self.history_object = None
        self.model = None
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x : x/255.0 - 0.5 , input_shape = self.inputShape))
        
        self.model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        
        self.model.add(Flatten())
        self.model.add(Dense(100, kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Dense(50, kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Dense(10, kernel_regularizer = self.kernalRegularizer, activation = self.activationMethod))
        self.model.add(Dense(1))        
        self.model.compile(loss = self.lossFunction, optimizer = self.optimizerMethod)

    def fit_model(self, oSettings):
        if self.model is None:
            self.build_model()
        self.history_object = self.model.fit(oSettings["X_train"], 
                                            oSettings["y_train"], 
                                            epochs = oSettings["epochs"], 
                                            verbose = 1, 
                                            validation_split = oSettings["validation_split"],
                                            validation_data = oSettings["validation_data"], 
                                            shuffle = True)

    def fit_generator(self, oSettings):
        if self.model is None:
            self.build_model()
        self.history_object = self.model.fit_generator(oSettings["train_generator"], 
                                    steps_per_epoch = oSettings["train_sample_len"], 
                                    epochs = oSettings["epochs"], 
                                    verbose = 1, 
                                    validation_data = oSettings["validation_generator"],
                                    validation_steps = oSettings["valid_sample_len"])
    def save(self, sFileName):
        self.model.save(sFileName)
    
    def visualize_loss(self):
        plt.plot(self.history_object.history['loss'])
        plt.plot(self.history_object.history['val_loss'])
        plt.title('Mean Squared Error Loss')
        plt.ylabel('mean squared error')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
