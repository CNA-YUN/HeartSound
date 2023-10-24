import os
import pandas as pd
import librosa
import librosa.display as display
import glob

import scipy
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

# def plot_signal(audio_data, title=None):
#     plt.figure(figsize=(12, 3.5), dpi=300)
#     plt.plot(audio_data, linewidth=1)
#     plt.title(title,fontsize = 16)
#     plt.tick_params(labelsize=12)
#     # plt.grid(axis='y')
#     plt.show()
# audio_path = r'F:/dataset/Cinc2016/total/a0002.wav'
# audio_data, fs = librosa.load(audio_path,sr=44100)
# plot_signal(audio_data, title='Initial Audio')

# normal_data=pd.read_csv(r'F:/dataset/Cinc2016/RECORDS-noproblem.csv',header=None)
# abnormal_data=pd.read_csv(r'F:/dataset/Cinc2016/RECORDS-problem.csv',header=None)
# normal_data=np.array(normal_data)
# abnormal_data=np.array(abnormal_data)

# # 两层LSTM
# model = Sequential()
# model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.35, return_sequences=True,input_shape = (100,1)))
# model.add(Conv1D(64, (3), padding="same",activation='relu'))
# model.add(MaxPooling1D((2), strides=(2)))
# model.add(BatchNormalization())
#
# model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# model.summary()

# CNN与LSTM结合
model = Sequential()

model.add(Conv1D(24, (15), padding="same", activation='relu', input_shape = (100, 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D((3), strides=(2)))

model.add(Conv1D(42, (7), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D((3), strides=(2)))

model.add(Conv1D(96, (3), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D((2), strides=(2)))

model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=True))
model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))
# model.add(LSTM(units=20, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))

model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

# 从csv文件读取数据和标签
data_file=r'csv/data_feature_baby.csv'
label_file=r'csv/label_baby.csv'
data=pd.read_csv(data_file)
data=np.array(data)
label=pd.read_csv(label_file)
label=np.array(label)

# 训练集与测试集划分
X_train,X_test,y_train,y_test=train_test_split(data,label,train_size=0.7,test_size=0.3,random_state=0)

# 训练模型
history=model.fit(X_train,y_train, batch_size=512, epochs=50,validation_split=0.1,verbose=1, shuffle=True)

# 使用测试集对模型进行测试
score=model.evaluate(X_test,y_test)
print("baby_score=",score)