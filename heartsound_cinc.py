import os
import pandas as pd
import librosa
import librosa.display as display
import glob
from tensorflow import keras
from scipy import signal
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D,Conv1D,BatchNormalization,ReLU,MaxPooling1D
from keras.optimizers import Adam
#from keras.utils import np_utils
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import samplerate
from sklearn.model_selection import train_test_split
def plot_signal(audio_data, title=None):
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(audio_data, linewidth=1)
    plt.title(title,fontsize = 16)
    plt.tick_params(labelsize=12)
    # plt.grid(axis='y')
    plt.show()
audio_path = 'F:/dataset/Cinc2016/total/a0002.wav'
audio_data, fs = librosa.load(audio_path,sr=44100)
#plot_signal(audio_data, title='Initial Audio')
normal_data=pd.read_csv('F:/dataset/Cinc2016/RECORDS-noproblem.csv',header=None)
abnormal_data=pd.read_csv('F:/dataset/Cinc2016/RECORDS-problem.csv',header=None)
normal_data=np.array(normal_data)
abnormal_data=np.array(abnormal_data)

def extract_data_cinc(folder):
    # function to load files and extract features
    file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    label=[]
    new_label=[]
    number=0
    for file_name in file_names:
        #audio_data=band_pass_filter(file_name,1,25,4000,fs)
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        down_sample_audio_data = samplerate.resample(X, 1000 / fs, converter_type='sinc_best')
        down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=down_sample_audio_data, sr=fs, n_mfcc=100).T,axis=0)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)
        print(number+1)
        number+=1
    for root,dirs,files in os.walk(folder):
        for file in files:
            if ".wav" in file:
                if file.replace(".wav",'') in abnormal_data:
                    label.append(0)
                else:
                    label.append(1)
    label=np.array(label).reshape([-1,1])
    data=np.array(data)
    return data,label

# 两层LSTM
model = Sequential()
model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.35, return_sequences=True,input_shape = (100,1)))
#model.add(Conv1D(64, (3), padding="same",activation='relu'))
#model.add(MaxPooling1D((2), strides=(2)))
model.add(BatchNormalization())

model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
'''
# CNN与LSTM结合
lstm_model = Sequential()

lstm_model.add(Conv1D(24, (15), padding="same",activation='relu',input_shape = (100,1)))
lstm_model.add(BatchNormalization())
lstm_model.add(ReLU())
lstm_model.add(MaxPooling1D((3), strides=(2)))

lstm_model.add(Conv1D(42, (7), padding="same",activation='relu'))
lstm_model.add(BatchNormalization())
lstm_model.add(ReLU())
lstm_model.add(MaxPooling1D((3), strides=(2)))

lstm_model.add(Conv1D(96, (3), padding="same",activation='relu'))
lstm_model.add(BatchNormalization())
lstm_model.add(ReLU())
lstm_model.add(MaxPooling1D((2), strides=(2)))

lstm_model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=True))
lstm_model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))
#lstm_model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))

lstm_model.add(BatchNormalization())
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
lstm_model.summary()
'''
data,label=extract_data_cinc('F:/dataset/Cinc2016/total')

# 训练集与测试集划分
X_train,X_test,y_train,y_test=train_test_split(data,label,train_size=0.7,test_size=0.3,random_state=0)

# 两层LSTM模型训练
history=model.fit(X_train,y_train, batch_size=512, epochs=40,validation_split=0.1,verbose=1, shuffle=True)
# 两层LSTM模型测试
score=model.evaluate(X_test,y_test)
print("score=",score)