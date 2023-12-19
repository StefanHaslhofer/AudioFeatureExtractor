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
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)

    plt.subplot(2, 2, 1)
    plt.plot(time, x, label='x-axis', color='r')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration')
    plt.title('x-axis')

    plt.subplot(2, 2, 2)
    plt.plot(time, y, label='y-axis', color='g')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration')
    plt.title('y-axis')

    plt.subplot(2, 2, 3)
    plt.plot(time, z, label='z-axis', color='b')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration')
    plt.title('z-axis')

    plt.subplot(2, 2, 4)
    plt.plot(time, total, label='total', color='c')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration')
    plt.title('total acceleration')

    plt.suptitle(label)
    plt.show()

    f = fft(plt_data.values[:, 4])
    freq = fftfreq(len(plt_data), d=1 / sr)
    plt.plot(freq[1:], abs(f)[1:])
    plt.title(label)
    plt.xlabel('freq (Hz)')
    plt.ylabel('power')
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

    # analyse frequency domain
    freq = fftfreq(len(sample), d=1 / sr)[1:]
    x_fft = abs(fft(sample[:, 1])[1:])
    y_fft = abs(fft(sample[:, 2])[1:])
    z_fft = abs(fft(sample[:, 3])[1:])
    mag_fft = abs(fft(sample[:, 4])[1:])

    # get dominant frequency
    max_x_freq = freq[np.argmax(x_fft)]
    max_y_freq = freq[np.argmax(y_fft)]
    max_z_freq = freq[np.argmax(z_fft)]
    max_mag_freq = freq[np.argmax(mag_fft)]

    # get dominant frequency energy
    max_x_en = np.max(x_fft)
    max_y_en = np.max(y_fft)
    max_z_en = np.max(z_fft)
    max_mag_en = np.max(mag_fft)

    x_en = np.sum(x_fft)
    y_en = np.sum(y_fft)
    z_en = np.sum(z_fft)
    mag_en = np.sum(mag_fft)

    q = np.quantile(x_fft, [0.25,0.75])
    q1_freq = q[0]
    q3_freq = q[1]

    return [mean_x, mean_y, mean_z, mean_mag, var_x, var_y, var_z, var_mag, max_x, max_y, max_z, max_mag,
            max_x_freq, max_y_freq, max_z_freq, max_mag_freq, max_x_en, max_y_en, max_z_en, max_mag_en,
            x_en, y_en, z_en, mag_en, q1_freq, q3_freq]


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
