import DataAquisition
import DataProcessing
import DataVisualization

from Training_Model import Model_Class

# select data source(s) here
use_my_data = False
use_udacity_data = True
use_track1 = False
use_track1_reverse = True
use_track1_corners = True
use_track2 = True

data_to_use = [use_my_data, 
               use_udacity_data, 
               use_track1, 
               use_track1_reverse, 
               use_track1_corners, 
               use_track2]

data_paths = ['../../Assets/data/myData/', 
              '../../Assets/data/sampleTrainingData/', 
              '../../Assets/data/Track1/', 
              '../../Assets/data/Track1Reverse/', 
              '../../Assets/data/Track1Corners/', 
              '../../Assets/data/Track2/']

image_paths, steering_angles = DataAquisition.get_dataSample_path(data_paths, data_to_use, bHasUdacityData = True)


image_paths, steering_angles = DataProcessing.data_normalization({
    'num_classes': 50, 
    'count_cutoff': int(len(steering_angles)/50),
    'image_paths': image_paths,
    'steering_angles': steering_angles
})

train_paths, valid_paths, train_angles, valid_angles = DataProcessing.data_split(image_paths, steering_angles, 0.2, 50)
X_train, y_train = DataProcessing.generate_processed_data(train_paths, train_angles)
X_valid, y_valid = DataProcessing.generate_processed_data(valid_paths, valid_angles, validation_flag = True)

# X_data, y_data = DataProcessing.generate_processed_data(image_paths, steering_angles)

oModel = Model_Class({
    'input_shape': (66, 200, 3),
    'l2_weight': 0.001,
    'activation': 'elu',
    'loss': 'mse',
    'optimizer': 'adam'
})
oModel.build_model()
oModel.fit_model({
    'X_train': X_train,
    'y_train': y_train,
    'epochs': 7,
    'validation_split': 0.2,
    'validation_data': (X_valid, y_valid)
})

oModel.save('model1.h5')
oModel.visualize_loss()


