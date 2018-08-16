import DataAquisition
import DataProcessing
import DataVisualization

from Training_Model import Model_Class

# Here the majority of the training data are captured from first track
# I included 2nd track to let the model at least be somewhat aware of the different terran
data_paths = ['../../Assets/data/sampleTrainingData/', 
              '../../Assets/data/Track1/', 
              '../../Assets/data/Track1Reverse/', 
              '../../Assets/data/Track1Corners/', 
              '../../Assets/data/Track2/']

# Get the data path to all the images
image_paths, steering_angles = DataAquisition.get_dataSample_path(data_paths, bHasUdacityData = True)

# Normalize the dataset to reduce the 'straight driving' portion thus the model is less biased towards straight driving
image_paths, steering_angles = DataProcessing.data_normalization({
    'num_classes': 50, 
    'count_cutoff': int(len(steering_angles)/100),
    'image_paths': image_paths,
    'steering_angles': steering_angles
})

# Get the training data and validation data
train_paths, valid_paths, train_angles, valid_angles = DataProcessing.data_split(image_paths, steering_angles, 0.1, 50)
X_train, y_train = DataProcessing.generate_processed_data(train_paths, train_angles)
X_valid, y_valid = DataProcessing.generate_processed_data(valid_paths, valid_angles, bValidation = True)

# X_data, y_data = DataProcessing.generate_processed_data(image_paths, steering_angles)

# Initialize all the parameters for the model
oModel = Model_Class({
    'input_shape': (66, 200, 3),
    'l2_weight': 0.001,
    'activation': 'elu',
    'loss': 'mse',
    'optimizer': 'adam'
})

# Build the traning model
oModel.build_model()

# Optimize the model, I did not specify the batch size because my computer can handle the currently given settings
oModel.fit_model({
    'X_train': X_train,
    'y_train': y_train,
    'epochs': 7,
    'validation_split': None,
    'validation_data': (X_valid, y_valid)
})

# Save the model for drive.py to use
oModel.save('model.h5')

# Visualize the loss
oModel.visualize_loss()


