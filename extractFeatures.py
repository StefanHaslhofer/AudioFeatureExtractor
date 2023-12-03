import librosa.feature
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

def slice_audio(audio, sr, labels):
    """
    get audio samples between start timestamp and end timestamp of labels
    """
    slices = []

    for idx in labels.index:
        # start/end is timestamp * sample rate
        start = round(labels['start'][idx] * sr)
        end = round(labels['end'][idx] * sr)
        slices.append(audio[start:end])

    return slices


def describe_mfcc(mfcc, sr):
    """
    extract features from audio range (as described in https://maelfabien.github.io/machinelearning/Speech9/#)
    """
    # mean value for each frequency bin
    bin_mean = np.mean(mfcc, axis=1)

    # mean energy
    mean = np.mean(mfcc)
    mean_squared = np.mean(mfcc) ** 2
    # standard derivation from mean energy
    std = np.std(mfcc)
    var = std ** 2
    median = np.median(mfcc)
    # mfcc bins with max/min mean energy
    max_energy = np.amax(bin_mean)
    max_energy_bin = np.argmax(bin_mean)
    min_energy = np.amin(bin_mean)
    min_energy_bin = np.argmin(bin_mean)
    # quantiles
    q1 = np.quantile(bin_mean, 0.25)
    q3 = np.quantile(bin_mean, 0.75)

    # larger vehicles should be asymmetric to lower frequencies
    skew = scipy.stats.skew(bin_mean)
    kurt = scipy.stats.kurtosis(bin_mean)
    iqr = scipy.stats.iqr(bin_mean)

    return [mean, mean_squared, std, var, median, max_energy, max_energy_bin, min_energy, min_energy_bin, q1, q3, skew, kurt, iqr]


def ext_freq_features(samples, sr, labels):
    """
    call feature vector extraction for all samples and map result to label vector
    """
    if len(samples) != len(labels):
        raise Exception('Samples must have same size as labels')

    vec = []
    for idx, y in enumerate(samples):
        print('extracting features of sample ', idx, ' of ', len(samples), '...')
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        # append tuple of feature vector and label vector
        vec.append((
            describe_mfcc(mfcc, sr),
            [labels['vehicleType'][idx], labels['direction'][idx]]
        ))

    return vec


audio, sample_rate = librosa.load('./data/processed/streetNoise1.wav')
labels = pd.read_csv('./data/processed/streetNoise1.csv', sep=";", header=0)

# extract features from audio range
audio_slices = slice_audio(audio, sample_rate, labels)
freq_features = ext_freq_features(audio_slices, sample_rate, labels)


def scatter_plot(data):
    plt.figure()
    plt.xlabel('energy')
    plt.ylabel('max_energy')

    x1 = []
    x2 = []
    for d in data:
        if d[1][0] == 'medium':
            x1.append(d[0][1])
            x2.append(d[0][5])

    plt.scatter(x1, x2)

    x1 = []
    x2 = []
    for d in data:
        if d[1][0] == 'heavy':
            x1.append(d[0][1])
            x2.append(d[0][5])

    plt.scatter(x1, x2)

    plt.show()


print(freq_features)
scatter_plot(freq_features)
