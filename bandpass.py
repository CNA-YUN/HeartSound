#测试巴特沃斯滤波器

import numpy as np
import samplerate
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import csv

filename=r'F:/dataset/Cinc2016/total/a0002.wav'
data=[]
#读取音频文件
audio_data, fs = librosa.load(filename,res_type='kaiser_fast')
#butter worth求滤波系数b和a
b,a=signal.butter(1,[2*25/fs,2*4000/fs],btype='bandpass')
# print(np.ndim(audio_data))
#滤波器滤波
new_signal=signal.lfilter(b,a,audio_data)
#下采样
down_sample_audio_data = samplerate.resample(new_signal, 1000 / fs, converter_type='sinc_best')
#归一化
down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
# 提取mfcc特征
mfccs = np.mean(librosa.feature.mfcc(y=down_sample_audio_data, sr=fs, n_mfcc=100).T,axis=0)
feature = np.array(mfccs)#.reshape([-1,1])
data.append(feature)
print("feature:",feature)
print("data_feature:",data)
# plt.plot(feature)
# plt.show()

csv_file= 'data_feature_cinc.csv'

with open(csv_file,'w',newline='')as csvfile:
    csv_writer=csv.writer(csvfile)
    for row in data:
        csv_writer.writerow(row)
print('特征值csv文件保存成功')
