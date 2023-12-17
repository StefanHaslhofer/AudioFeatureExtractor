import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

gen_plt = True
# I chose a low sampling rate if 100Hz because I do not need information of higher frequencies.
# For this task in particular we can assume that hand and foot movements stay similar and minor changes
# in higher frequencies can be ignored (the human body is not able to swing arms/feet hundreds of times per second).
sr = 100


def plot_movement(time, x, y, z, total, label):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(time, x, label='x-axis', color='r')
    plt.title('x-axis')

    plt.subplot(2, 2, 2)
    plt.plot(time, y, label='y-axis', color='g')
    plt.title('y-axis')

    plt.subplot(2, 2, 3)
    plt.plot(time, z, label='z-axis', color='b')
    plt.title('z-axis')

    plt.subplot(2, 2, 4)
    plt.plot(time, total, label='total', color='c')
    plt.title('total acceleration')

    plt.suptitle(label)
    plt.show()


def slice_sample(sample, sr, T):
    """
    split movement sample into segments of size n
    """
    n = sr * T
    for i in range(0, len(sample), n):
        yield sample[i:i + n]


def describe_sample(sample):
    """
    extract features from single movement sample slice
    """
    mean = np.mean(abs(sample), axis=0)
    mean_x = mean[1]
    mean_y = mean[2]
    mean_z = mean[3]
    mean_mag = mean[4]

    var = np.var(sample, axis=0)
    var_x = var[1]
    var_y = var[2]
    var_z = var[3]
    var_mag = var[4]

    # absolute maximum acceleration
    max_acc = np.max(abs(sample), axis=0)
    max_x = max_acc[1]
    max_y = max_acc[2]
    max_z = max_acc[3]
    max_mag = max_acc[4]

    x_stft = fft(sample[:, 1])
    y_stft = fft(sample[:, 2])
    z_stft = fft(sample[:, 3])
    mag_stft = fft(sample[:, 4])

    return [mean_x, mean_y, mean_z, mean_mag, var_x, var_y, var_z, var_mag, max_x, max_y, max_z, max_mag]


def extract_features(samples, label):
    vec = []
    for idx, sample in enumerate(samples):
        print('extracting features of sample ', idx, ' of ', len(samples), '...')
        # append tuple of feature vector and label vector
        vec.append((
            describe_sample(sample),
            label
        ))

    return vec


# plot left hand
left_hand = pd.read_csv('data/raw/leftHand.csv', sep=",", header=0)
if gen_plt:
    plt_data = left_hand[1000:1500]
    plot_movement(plt_data['time'], plt_data['ax (m/s^2)'], plt_data['ay (m/s^2)'], plt_data['az (m/s^2)'],
                  plt_data['aT (m/s^2)'], 'left hand')

# plot right hand
right_hand = pd.read_csv('data/raw/rightHand.csv', sep=",", header=0)
if gen_plt:
    plt_data = right_hand[1000:1500]
    plot_movement(plt_data['time'], plt_data['ax (m/s^2)'], plt_data['ay (m/s^2)'], plt_data['az (m/s^2)'],
                  plt_data['aT (m/s^2)'], 'right hand')

# plot left pocket
left_pocket = pd.read_csv('data/raw/leftPocket.csv', sep=",", header=0)
if gen_plt:
    plt_data = left_pocket[1000:1500]
    plot_movement(plt_data['time'], plt_data['ax (m/s^2)'], plt_data['ay (m/s^2)'], plt_data['az (m/s^2)'],
                  plt_data['aT (m/s^2)'], 'left pocket')

# plot right pocket
right_pocket = pd.read_csv('data/raw/rightPocket.csv', sep=",", header=0)
if gen_plt:
    plt_data = right_pocket[1000:1500]
    plot_movement(plt_data['time'], plt_data['ax (m/s^2)'], plt_data['ay (m/s^2)'], plt_data['az (m/s^2)'],
                  plt_data['aT (m/s^2)'], 'right pocket')

# split samples into slices of 5 sec
slices = list(slice_sample(left_hand.values, sr=sr, T=5))
feat = extract_features(slices, 'left_hand')
