import csv
import cv2
import os
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def get_driveLogs(csv_path):
    """Public: get the driving log 
        
    @Param:
    csv_path: dir that contains driving logs;"""
    csvLogs = []
    with open(csv_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)

        #skip the first line
        next(reader)

        for line in reader:
            csvLogs.append(line)
    return csvLogs

def get_samples(data_path, lines):
    """Public: get data samples  
        
    @Param:
    data_path: dir that contains driving image data;
    lines: csv log files that contains the information for the image data"""

    img_path = data_path + 'IMG/'
    images = []
    steering_angles = []
 
    for line in lines:
        steering_angle = float(line[3])
        steering_correction = 0.2
 
        for i in range(3):
            source_path = line[i]
            filename = source_path.split(os.sep)[-1]
            # filename = source_path.split('/')[-1]
            # originalImage = cv2.imread(img_path + filename)
            # image = originalImage[65:135,:,:]
            # image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
             
            image = cv2.imread(img_path + filename)
            images.append(image)
 
            if i == 1:
                steering_angles.append(steering_angle + steering_correction)
            elif i == 2:
                steering_angles.append(steering_angle - steering_correction)
            else:
                steering_angles.append(steering_angle)
 
                if steering_angle > 0.33:
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    steering_angles.append( -1 * steering_angle )
     
    return images, steering_angles

def data_split(lines, valid_size):
    return train_test_split(lines, test_size = valid_size)

def get_batchLine_data(data_path, batch_line):
    img_path = data_path + 'IMG/'

    steering_angle = np.float32(batch_line[3])
    images, steering_angles = [], []
 
    for i in range(3):
        source_path = batch_line[i]
        filename = source_path.split(os.sep)[-1]

        image = cv2.imread(img_path + filename)
        images.append(image)
 
        steering_correction = 0.2
        if i == 1:
            steering_angles.append(steering_angle + steering_correction)
        elif i == 2:
            steering_angles.append(steering_angle - steering_correction)
        else:
            steering_angles.append(steering_angle)
 
            if steering_angle > 0.33:
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                steering_angles.append( -1 * steering_angle )
 
    return images, steering_angles

def data_generator(data_path, samples, batch_size=128):
    num_samples = len(samples)
 
    while True:
        shuffle(samples)
 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, steering_angles = [], []
 
            for batch_sample in batch_samples:
                augmented_images, augmented_angles = get_batchLine_data(data_path, batch_sample)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)
 
            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)