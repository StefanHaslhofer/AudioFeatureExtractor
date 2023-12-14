import librosa.feature
import numpy as np
import pandas as pd
import scipy


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


def describe_mfcc(mfcc):
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
    # quantiles and mean bin
    q1 = np.quantile(bin_mean, 0.25)
    q3 = np.quantile(bin_mean, 0.75)

    # larger vehicles should be asymmetric to lower frequencies
    skew = scipy.stats.skew(bin_mean)
    iqr = scipy.stats.iqr(bin_mean)

    return [mean, mean_squared, std, var, median, max_energy, max_energy_bin, min_energy, min_energy_bin, q1, q3, skew, iqr]


def ext_freq_features(samples, sr, labels):
    """
    call feature vector extraction for all samples and map result to label vector
    """
    if len(samples) != len(labels):
        raise Exception('Samples must have same size as labels')

    vec = []
    for idx, y in enumerate(samples):
        print('extracting features of sample ', idx, ' of ', len(samples), '...')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # append tuple of feature vector and label vector
        vec.append((
            describe_mfcc(mfcc),
            [labels['vehicleType'][idx], labels['direction'][idx]]
        ))

    return vec


def write_arff(freq_features):
    f = open('data/processed/drivingDirection_data.arff', 'w')
    # write header
    f.write('@RELATION vehicle\n\n')

    attributes = ['mean', 'mean_squared', 'std', 'var', 'median', 'max_energy', 'max_energy_bin', 'min_energy',
                  'min_energy_bin', 'q1', 'q3', 'skew', 'iqr']
    for a in attributes:
        f.write('@ATTRIBUTE ' + a + ' NUMERIC\n')
    vehicle_types = ['medium', 'heavy']
    f.write('@ATTRIBUTE vehicle_type {' + ','.join(vehicle_types) + '}\n')
    directions = ['LR', 'RL']
    f.write('@ATTRIBUTE direction {' + ','.join(directions) + '}\n')
    f.write('\n')

    # write data
    f.write('@DATA\n')
    for feats in freq_features:
        f.write(','.join(str(n) for n in feats[0]) + ',' + ','.join(feats[1]) + '\n')

    f.close()


audio, sample_rate = librosa.load('data/processed/drivingDirection_audio_raw.wav')
labels = pd.read_csv('data/processed/drivingDirection_labels.csv', sep=";", header=0)

# extract features from audio range
audio_slices = slice_audio(audio, sample_rate, labels)
freq_features = ext_freq_features(audio_slices, sample_rate, labels)

write_arff(freq_features)
