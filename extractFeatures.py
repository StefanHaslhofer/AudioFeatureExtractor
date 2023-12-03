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


def describe_stft(stft, sr):
    """
    extract features from audio range (as described in https://maelfabien.github.io/machinelearning/Speech9/#)
    """
    db = librosa.amplitude_to_db(np.mean(stft, axis=1), ref=np.max)
    librosa.display.specshow(db, sr=sr, y_axis='log', x_axis='time')
    plt.show()

    # mean value for each frequency bin
    freq_mean = np.mean(stft, axis=1)
    freq = librosa.fft_frequencies(sr=sr)

    # mean energy
    mean = np.mean(stft)
    energy = np.mean(stft) ** 2
    # standard derivation from mean energy
    std = np.std(stft)
    var = std ** 2
    # frequencies with max/min mean energy
    max_energy = np.amax(freq_mean)
    max_energy_mfcc = freq[np.argmax(freq_mean)]
    min_energy = np.amin(freq_mean)
    min_energy_mfcc = freq[np.argmin(freq_mean)]
    # quantiles
    q1 = np.quantile(freq_mean, 0.25)
    q3 = np.quantile(freq_mean, 0.75)
    # frequency bin of each quantile
    q1_mfcc = np.where(freq_mean == q1)[0][0]
    q2_mfcc = np.where(freq_mean == q3)[0][0]

    median = np.median(stft)
    # larger vehicles should be asymmetric to lower frequencies
    skew = scipy.stats.skew(freq_mean)
    kurt = scipy.stats.kurtosis(freq_mean)
    iqr = scipy.stats.iqr(freq_mean)

    return [mean, energy, std, max_energy, max_energy_mfcc, min_energy, min_energy_mfcc, q1, q3, q1_mfcc, q2_mfcc, median, skew, kurt, iqr]


def ext_freq_features(samples, sr, labels):
    """
    call feature vector extraction for all samples and map result to label vector
    """
    if len(samples) != len(labels):
        raise Exception('Samples must have same size as labels')

    vec = []
    for idx, y in enumerate(samples):
        print('extracting features of sample ', idx, ' of ', len(samples), '...')
        stft_abs = np.abs(librosa.stft(y))
        # append tuple of feature vector and label vector
        vec.append((
            describe_stft(stft_abs, sr),
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
    plt.ylabel('mean')

    x1 = []
    x2 = []
    for d in data:
        if d[1][0] == 'Car':
            x1.append(d[0][0])
            x2.append(d[0][1])

    plt.scatter(x1, x2)

    x1 = []
    x2 = []
    for d in data:
        if d[1][0] == 'Bus':
            x1.append(d[0][0])
            x2.append(d[0][1])

    plt.scatter(x1, x2)


    x1 = []
    x2 = []
    for d in data:
        if d[1][0] == 'Truck':
            x1.append(d[0][0])
            x2.append(d[0][1])

    plt.scatter(x1, x2)
    plt.show()


print(freq_features)
scatter_plot(freq_features)
