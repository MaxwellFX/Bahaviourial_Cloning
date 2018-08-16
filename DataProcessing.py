import cv2
import os
import numpy as np


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def data_split(lines, valid_size):
    return train_test_split(lines, test_size = valid_size)

def get_steering_angle_distribution(steering_angles, num_classes):
    return np.histogram(steering_angles, num_classes)

def data_normalization(oSettings):
    num_classes = oSettings["num_classes"]
    count_cutoff = oSettings["count_cutoff"]
    image_paths = oSettings["image_paths"]
    steering_angles = oSettings["steering_angles"]

    histogram, thresholds = get_steering_angle_distribution(steering_angles, num_classes)

    keep_probability = []
    for i in range(num_classes):
        if histogram[i] < count_cutoff:
            keep_probability.append(1.)
        else:
            keep_probability.append(1./(histogram[i]/count_cutoff))
    
    index_to_remove = []
    for i in range(len(steering_angles)):
        for j in range(num_classes):
            if steering_angles[i] > thresholds[j] and steering_angles[i] <= thresholds[j+1]:
                # delete from X and y with probability 1 - keep_probability[j]
                if np.random.rand() > keep_probability[j]:
                    index_to_remove.append(i)
    image_paths = np.delete(image_paths, index_to_remove, axis=0)
    steering_angles = np.delete(steering_angles, index_to_remove)

    return image_paths, steering_angles

def process_img(img):
    new_img = img[50:140,:,:]

    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)

    # scale to 66x200x3 
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
   
    # convert to YUV color space according to nVidia
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def rand_brightness(img):
    pixel_bias = np.random.randint(-28, 28)

    # preventing pixel value going beyond [0,255] range
    if pixel_bias > 0:
        mask = (img[:,:,0] + pixel_bias) > 255 
    else:
        mask = (img[:,:,0] + pixel_bias) < 0
    img[:,:,0] += np.where(mask, 0, pixel_bias)

    return img

def rand_dim(img):
    cols = img.shape[1]
    mid = np.random.randint(0, cols)
    factor = np.random.uniform(0.6, 0.8)

    # randomly dim one side of the image
    if np.random.rand() > 0.5:
        img[:, 0 : mid, 0] *= factor
    else:
        img[:, mid : cols, 0] *= factor
    
    return img

def rand_shift(img):
    rows, cols = img.shape[0:2]
    horizon = 2 * rows /5 # works for this particular image size (66, 200, 3)
    vertical_shift = np.random.randint(-rows/8, rows/8)

    pts1 = np.float32([[0, horizon],
                       [cols, horizon],
                       [0, rows],
                       [cols, rows]])

    pts2 = np.float32([[0, horizon + vertical_shift],
                       [cols, horizon + vertical_shift],
                       [0, rows],
                       [cols, rows]])

    transform_matrix = cv2.getPerspectiveTransform(pts1,pts2)

    return cv2.warpPerspective(img, transform_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

def image_agumentation(img):
    # covert pixel value to float for data augmentation
    augmented_img = img.astype(float)
    augmented_img = rand_shift(rand_dim(rand_brightness(augmented_img)))

    return augmented_img.astype(np.uint8)

def generate_processed_data(image_paths, steering_angles, validation_flag = False):
    image_paths, steering_angles = shuffle(image_paths, steering_angles)
    new_images = []
    new_angles = []

    for i in range(len(steering_angles)):
        img = cv2.imread(image_paths[i])
        angle = steering_angles[i]
        img = process_img(img)

        if not validation_flag:
            img = image_agumentation(img)

        new_images.append(img)
        new_angles.append(angle)

        # flip horizontally and invert steer angle, if magnitude is > 0.33
        if abs(angle) > 0.33:
            img = cv2.flip(img, 1)
            angle *= -1
            new_images.append(img)
            new_angles.append(angle)
    
    new_images = np.array(new_images)
    new_angles = np.array(new_angles)

    return new_images, new_angles     

# NOTE: Most likely won't using these because running fit_generator is miserably slow on my local machine
def _retrive_batchLine_data(data_path, batch_line):
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

# NOTE: Most likely won't using these because running fit_generator is miserably slow on my local machine
def data_generator(data_path, samples, batch_size=128):
    num_samples = len(samples)
 
    while True:
        shuffle(samples)
 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, steering_angles = [], []
 
            for batch_sample in batch_samples:
                augmented_images, augmented_angles = _retrive_batchLine_data(data_path, batch_sample)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)
 
            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)