# HeartSound
婴幼儿心音，通过深度学习方法判断心脏是否病变

## 2023.10.24
- 数据集Cinc2016，采集mfcc系数100个，LSTM神经网络，得到测试集cinc_score= [0.23585250973701477, 0.9300411343574524]
- 数据集Cinc2016，采集mfcc系数100个，cnn+LSTM神经网络，得到测试集cinc_score= [0.2707505524158478, 0.9248971343040466]
- 数据集baby，采集mfcc系数100个，LSTM网络，得到测试集baby_score= [0.6791784167289734, 0.6037735939025879]
- 数据集baby，采集mfcc系数100个，cnn+LSTM网络，得到测试集baby_score= [0.6682189106941223, 0.698113203048706]
## 2023.10.25
- 数据集baby，采集mfcc系数128个，cnn+lstm网络，得到测试集baby_score=[0.6572191119194031, 0.6037735939025879]
- 数据集baby，采集mfcc系数128个，lstm网络，得到测试集baby_score=[0.6779104471206665, 0.6037735939025879]
