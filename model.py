import numpy as np
import dataAcquisition

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

from keras.models import Model
import matplotlib.pyplot as plt


GIVEN_DATA_PATH = '../../Assets/data/sampleTrainingData/'
MY_DATA_PATH = '../../Assets/data/myData/'

ACTING_PATH = MY_DATA_PATH

lines = dataAcquisition.get_driveLogs(ACTING_PATH)

images, steering_angles = dataAcquisition.get_samples(ACTING_PATH, lines)
X_train = np.array(images)
y_train = np.array(steering_angles)
 
# traningSample, validationData = dataAcquisition.data_split(lines, 0.2)
# train_generator = dataAcquisition.data_generator(ACTING_PATH, traningSample, 128)
# validation_generator = dataAcquisition.data_generator(ACTING_PATH, validationData, 128)

#======================================
model = Sequential()
model.add(Lambda(lambda x : x/255.0 - 0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
 
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#======================================

history_object = model.fit(X_train, y_train, epochs = 3, verbose = 1, validation_split = 0.2, shuffle = True)

# history_object = model.fit_generator(train_generator, 
#                                     steps_per_epoch = len(traningSample), 
#                                     epochs = 3, 
#                                     verbose = 1, 
#                                     validation_data = validation_generator,
#                                     validation_steps = len(validationData))
 
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Squared Error Loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()