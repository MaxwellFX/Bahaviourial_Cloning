import cv2
import os
import numpy as np


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def data_split(X, y, valid_size, randomState):
    return train_test_split(X, y, test_size = valid_size, random_state = randomState)

def get_steering_angle_distribution(steering_angles, num_classes):
    return np.histogram(steering_angles, num_classes)

def data_normalization(oSettings):
    """normalize the dataset to reduce the 'straight driving' data, this way the model is not skew towards'straight driving' """
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
    
    indices_to_remove = []
    for i in range(len(steering_angles)):
        for j in range(num_classes):
            if steering_angles[i] > thresholds[j] and steering_angles[i] <= thresholds[j+1]:
                if np.random.rand() > keep_probability[j]:
                    indices_to_remove.append(i)
    image_paths = np.delete(image_paths, indices_to_remove, axis=0)
    steering_angles = np.delete(steering_angles, indices_to_remove)

    return image_paths, steering_angles

def process_img(img):
    '''pre-process the image to better fit to the nVidia training model, note that the color space is converted to YUV as suggested by nVidia'''

    #only keep a portion of the image that matters to the decision
    new_img = img[50:135,:,:]

    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)

    return new_img

def rand_brightness(img):
    '''apply random brightness to the image'''
    pixel_bias = np.random.randint(-30, 30)

    # preventing pixel value going beyond [0,255] range
    if pixel_bias > 0:
        mask = (img + pixel_bias) > 255 
    else:
        mask = (img + pixel_bias) < 0
    
    return img + np.where(mask, 0, pixel_bias)


def rand_dim(img):
    '''randomly dim a horizontal portion of the image'''
    cols = img.shape[1]
    mid = np.random.randint(0, cols)
    factor = np.random.uniform(0.6, 0.8)

    if np.random.rand() > 0.5:
        img[:, 0 : mid, :] *= factor
    else:
        img[:, mid : cols, :] *= factor
    
    return img

def rand_shift(img):
    '''randomly shift the horizon up or down'''
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
    '''randomly augment the data to create more variaty of the training image'''
    # covert pixel value to float for data augmentation
    augmented_img = img.astype(float)
    augmented_img = rand_shift(rand_dim(rand_brightness(augmented_img)))

    return augmented_img.astype(np.uint8)

def generate_processed_data(image_paths, steering_angles, bValidation = False):
    image_paths, steering_angles = shuffle(image_paths, steering_angles)
    new_images = []
    new_angles = []

    if bValidation:
        print("processing validation data...")
    else:
        print("processing training data..., this could take a while")
    for i in range(len(steering_angles)):
        img = cv2.imread(image_paths[i])
        angle = steering_angles[i]
        img = process_img(img)

        if not bValidation:
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

    print("Data preprocessing finished!")
    return new_images, new_angles