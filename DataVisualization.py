import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_steering_historgram(viz_settings):
    num_bins = viz_settings["num_bins"]
    steering_angles = viz_settings["steering_angles"]
    count_cutoff = viz_settings["count_cutoff"]

    hist, bins = np.histogram(steering_angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(steering_angles), np.max(steering_angles)), (count_cutoff, count_cutoff), 'k-')
    plt.show()
