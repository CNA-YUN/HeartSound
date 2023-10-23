import os
import librosa
import librosa.display as display
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import keras.utils
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile
plt.figure(figsize=(12, 4))
def band_pass_filter(original_signal, order, fc1,fc2, fs):
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

data, sampling_rate = soundfile.read('E://dataset//normal//normal__103_1305031931979_B.wav')
display.waveshow(data, sr=sampling_rate)


def extract_data(folder,fs):
    # function to load files and extract features
    file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        file_name=band_pass_filter(file_name,1,25,400,fs)
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data


normal_onehot = 1
murmur_onehot = 0

normal_sounds = extract_data("E://dataset//normal",sampling_rate)
normal_labels = [normal_onehot for items in normal_sounds]
murmur_sounds = extract_data("E://dataset//murmur",sampling_rate)
x_test = extract_data("E://dataset//test",sampling_rate)
murmur_labels = [murmur_onehot for items in murmur_sounds]
x_train = np.concatenate((normal_sounds, murmur_sounds))
y_train = np.concatenate((normal_labels, murmur_labels))

print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(40, 1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100)
model.summary()
