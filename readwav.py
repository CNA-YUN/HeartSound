# 使用librosa或者scipy库读取wav音频文件

import scipy
import pandas as pd
import matplotlib.pyplot as plt
import librosa
samplingrate,data=scipy.io.wavfile.read('F:/dataset/Cinc2016/total/a0012.wav')
print("sampling rate = {} Hz, length = {} samples,dtype = {}".format(samplingrate,*data.shape,data.dtype))
x,sr=librosa.load('F:/dataset/Cinc2016/total/a0012.wav',sr=44100)
print(x)
# plt.plot(x)
# plt.show()
