# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
import NeuralNetwork


nn = NeuralNetwork.NeuralNetwork()

#---------------------------データの前処理------------------------------
dir_here =  os.path.dirname(os.path.abspath(__file__))
path = dir_here + '/csv/correct_data.csv'
data = nn.str2int(path)[0]
# ノイズデータの除去
noize_indexes = np.array([index for index,num in enumerate(data) if num==-1 ])
# 正解データを作成
correct = np.delete(data, noize_indexes)

# 学習データの作成
path = dir_here + '/csv/pose.csv'
data = nn.str2int(path)
input_data = np.delete(data, noize_indexes, 0)

n_data = len(correct)
clusta_Num = len(set(correct))

print('データ数：' + str(n_data))
print('クラスタ数：' + str(clusta_Num))
print('学習データの形状：' + str(np.shape(input_data)))
print('正解データの形状：' + str(np.shape(correct)))

#標準化する
#def standardization(array):
    #ave_input = np.average(array, axis=0)
    #std_input = np.std(array, axis=0)
    #standardized = (array - ave_input) / std_input 
    #return standardized

#正解データをone-hot表現にする
correct_data = np.zeros((n_data,clusta_Num)) #[0,0,0]をn_data個 (n_datax3の行列)
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0
    #correct_dataのi番目の要素(i番目の行)の中の
    #corcorrectrect[i]番目(0番目,1番目,2番目)の要素を1に変換する
print('correct_data --> {}'.format(np.shape(correct_data)))

#訓練データとテストデータに分割    
index = np.arange(n_data)
index_train = index[index%2 == 0] # == [0 2 4 ... ]
index_test = index[index%2 != 0]  # == [1 3 5 ... ]

input_train = []
input_test = []
correct_train = []
correct_test = []

for i in range(n_data):
    if i%2==0:
        input_train.append(input_data[i])
        correct_train.append(correct_data[i])
for i in range(n_data):
    if i%2!=0:
        input_test.append(input_data[i])
        correct_test.append(correct_data[i])
input_train = np.array(input_train)
input_test = np.array(input_test)
correct_train = np.array(correct_train)
correct_test = np.array(correct_test)

n_train = input_train.shape[0] #訓練用データのサンプル数
n_test = input_test.shape[0]    #テスト用データのサンプル数
#---------------------------------------------------------------------------------

#NNの定義
n_in = np.shape(input_data)[1]
n_mid = 100
n_out = clusta_Num



# -- 各層の初期化 --
ml_1 = nn.MiddleLayer(n_in, n_mid)
dp_1 = nn.Dropout(0.5)
ml_2 = nn.MiddleLayer(n_mid, n_mid)
dp_2 = nn.Dropout(0.5)
ol = nn.OutputLayer(n_mid, n_out)
# -- 誤差の記録用 --
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

# -- 学習と経過の記録 --
n_batch = n_train // batch_size  # 1エポックあたりのバッチ数
for i in range(epoch):

    # -- 誤差の計測 --  
    fp(input_train, False)
    error_train = get_error(correct_train, n_train)
    fp(input_test, False)
    error_test = get_error(correct_test, n_test)
    
    # -- 誤差の記録 -- 
    test_error_x.append(i)
    test_error_y.append(error_test) 
    train_error_x.append(i)
    train_error_y.append(error_train) 
    
    # -- 経過の表示 -- 
    if i%interval == 0:
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))

    # -- 学習 -- 
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)  # インデックスをシャッフルする
    for j in range(n_batch):
        
        # ミニバッチを取り出す
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]
        
        # 順伝播と逆伝播
        fp(x, True)
        bp(t)
        
        # 重みとバイアスの更新
        uppdate_wb() 


#誤差の記録をグラフ表示
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")


#学習した重みを保存
w_ml_1 = dir_here + '/csv/NN-result/w_ml_1.csv'
b_ml_1 = dir_here + '/csv/NN-result/b_ml_1.csv'
w_ml_2 = dir_here + '/csv/NN-result/w_ml_2.csv'
b_ml_2 = dir_here + '/csv/NN-result/b_ml_2.csv'
w_ol = dir_here + '/csv/NN-result/w_ol.csv'
b_ol = dir_here + '/csv/NN-result/b_ol.csv'
        
with open(w_ml_1, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(ml_1.w)
with open(b_ml_1, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerow(ml_1.b)
with open(w_ml_2, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(ml_2.w)
with open(b_ml_2, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerow(ml_2.b)
with open(w_ol, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(ol.w)
with open(b_ol, 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerow(ol.b)



#正解率の計算
fp(input_train,False)
count_train = np.sum(np.argmax(ol.y,axis=1) == np.argmax(correct_train, axis=1))

fp(input_test,False)
count_test = np.sum(np.argmax(ol.y,axis=1) == np.argmax(correct_test, axis=1))

print('学習データでの正解数---{}/{}'.format(count_train,n_train))
print('テストデータでの正解数---{}/{}'.format(count_test,n_test))

print("Accuracy Train:" + str(float(count_train)/n_train*100) + "%",
      "Accuracy Test:" + str(float(count_test)/n_test*100) + "%")