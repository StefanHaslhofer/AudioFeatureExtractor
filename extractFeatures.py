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


def describe_freq(freq):
    """
    extract features from audio range (as described in https://maelfabien.github.io/machinelearning/Speech9/#)
    """
    mean = np.mean(freq)
    std = np.std(freq)
    maxv = np.amax(freq)
    minv = np.amin(freq)
    median = np.median(freq)
    # larger vehicles should be asymmetric to lower frequencies
    skew = scipy.stats.skew(freq)
    kurt = scipy.stats.kurtosis(freq)
    q1 = np.quantile(freq, 0.25)
    q3 = np.quantile(freq, 0.75)
    iqr = scipy.stats.iqr(freq)
    energy = np.sum(freq ** 2)

    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, iqr, energy]


def ext_freq_features(samples, labels):
    """
    call feature vector extraction for all samples and map result to label vector
    """
    if len(samples) != len(labels):
        raise Exception('Samples must have same size as labels')

    vec = []
    for idx, x in enumerate(samples):
        freqs = np.fft.fftfreq(x.size)
        # append tuple of feature vector and label vector
        vec.append((
            describe_freq(freqs),
            [labels['vehicleType'][idx], labels['direction'][idx]]
        ))

    return vec


audio, sample_rate = librosa.load('./data/processed/streetNoise1.wav')
labels = pd.read_csv('./data/processed/streetNoise1.csv', sep=";", header=0)

# extract features from audio range
audio_slices = slice_audio(audio, sample_rate, labels)
freq_features = ext_freq_features(audio_slices, labels)


def scatter_plot(data):
    plt.figure()

    plt.scatter(data[data[:][1][0] == 'Car'][0][0], data[data[:][1][0] == 'Car'][0][10],)
    plt.show()

print(freq_features)
scatter_plot(freq_features)


