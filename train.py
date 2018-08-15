import numpy as np
import dataAcquisition
from Training_Model import Model_Class
import matplotlib.pyplot as plt


GIVEN_DATA_PATH = '../../Assets/data/sampleTrainingData/'
MY_DATA_PATH = '../../Assets/data/myData/'

ACTING_PATH = MY_DATA_PATH

lines = dataAcquisition.get_driveLogs(ACTING_PATH)

images, steering_angles = dataAcquisition.get_samples(ACTING_PATH, lines)
X_train = np.array(images)
y_train = np.array(steering_angles)
 
# trainingSample, validationData = dataAcquisition.data_split(lines, 0.2)
# train_generator = dataAcquisition.data_generator(ACTING_PATH, trainingSample, 128)
# validation_generator = dataAcquisition.data_generator(ACTING_PATH, validationData, 128)

oModel = Model_Class({
    'input_shape': (160, 320, 2),
    'crop_range': (65, 25),
    'l2_weight': 0.001,
    'activation': 'elu',
    'loss': 'mse',
    'optimizer': 'adam'
})
oModel.build_model()
oModel.fit_model({
    'X_train': X_train,
    'y_train': y_train,
    'epochs': 3,
    'validation_split': 0.2,
})

# oModel.fit_generator({
#     'train_generator': train_generator,
#     'train_sample_len': len(trainingSample),
#     'epochs': 3,
#     'validation_generator': validation_generator,
#     'valid_sample_len': len(validationData)
# })

oModel.save('model.h5')
oModel.visualize_loss()


