# 使用pandas库读取csv文件后用numpy库转换为数组形式便于输入神经网络

import csv
import pandas
import numpy
from scipy import signal

datafile=r'data_feature_cinc.csv'
labelfile=r'label_cinc.csv'

data=pandas.read_csv(datafile)
data=numpy.array(data)
label=pandas.read_csv(labelfile)
label=numpy.array(label)

print("data.shape:",data.shape)
print("label.shape:",label.shape)