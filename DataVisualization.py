import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_steering_historgram(viz_settings):
    num_bins = viz_settings["num_bins"]
    steering_angles = viz_settings["steering_angles"]
    avg_samples = viz_settings["avg_samples_per_bin"]

    hist, bins = np.histogram(steering_angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(steering_angles), np.max(steering_angles)), (avg_samples, avg_samples), 'k-')
    plt.show()