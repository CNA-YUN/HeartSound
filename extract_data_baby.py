import glob
import os
import librosa
import numpy as np
import pandas as pd
import samplerate
from scipy import signal
import csv

normal_data=pd.read_csv(r'F:\dataset\baby\RECORDS-normal.csv',header=None)
abnormal_data=pd.read_csv(r'F:\dataset\baby\RECORDS-abnormal.csv',header=None)
normal_data=np.array(normal_data)
abnormal_data=np.array(abnormal_data)

# print(normal_data)
def band_pass_filter(original_signal, order, fc1,fc2, fs):
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

folder=R'F:\dataset\baby\infants'
folder=os.path.normpath(folder)
file_names = glob.glob(os.path.join(folder, '*.wav'))
data = []
label=[]
number=0

# print(file_names)

for file_name in file_names:
    X, fs = librosa.load(file_name, res_type='kaiser_fast')

    audio_data=band_pass_filter(X,1,25,4000,fs)

    # here kaiser_fast is a technique used for faster extraction
    down_sample_audio_data = samplerate.resample(audio_data, 1000 / fs, converter_type='sinc_best')

    down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))

    # we extract mfcc feature from data
    mfccs = np.mean(librosa.feature.mfcc(y=down_sample_audio_data, sr=fs, n_mfcc=100).T,axis=0)
    feature = np.array(mfccs)
    data.append(feature)
    print(number+1,"/528")
    number+=1
for root,dirs,files in os.walk(folder):

    # print(files)
    for file in files:
        file_replace=file.replace(".wav",'')
        print(np.array(int(file_replace)))
        if np.array(int(file_replace)) in abnormal_data:
            label.append(0)
        else:
            label.append(1)

# print(label)

csv_file= 'csv/label_baby.csv'
label_csv=[[item]for item in label]
with open(csv_file,'w',newline='')as csvfile:
    csv_writer=csv.writer(csvfile)
    csv_writer.writerows(label_csv)
print('label_csv保存成功')


# print(abnormal_data)
# print(label.count(0))

csv_file= 'data_feature_baby.csv'
with open(csv_file,'w',newline='')as csvfile:
    csv_writer=csv.writer(csvfile)
    for row in data:
        csv_writer.writerow(row)
print('data_baby文件保存成功')