# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
from load_data import LoadData
import torch
from sklearn.model_selection import train_test_split
import json

# -- 各層の継承元 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)  # 重み（行列）
        self.b = wb_width * np.random.randn(n)  # バイアス（ベクトル）

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8
        
    def update(self, eta):      
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)  # ReLU
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLUの微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 

# -- 出力層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # ソフトマックス関数

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
        
# -- ドロップアプト --
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio  # ニューロンを無効にする確率

    def forward(self, x, is_train):  # is_train: 学習時はTrue
        if is_train:
            rand = np.random.rand(*x.shape)  # 入力と同じ形状の乱数の行列
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)  # 1:有効 0:無効
            self.y = x * self.dropout  # ニューロンをランダムに無効化
        else:
            self.y = (1-self.dropout_ratio)*x  # テスト時は出力を下げる
        
    def backward(self, grad_y):
        self.grad_x = grad_y * self.dropout  # 無効なニューロンでは逆伝播しない

# -- 順伝播 --
def fp(x, is_train):
    ml_1.forward(x)
    dp_1.forward(ml_1.y, is_train)
    ml_2.forward(dp_1.y)
    dp_2.forward(ml_2.y, is_train)
    ol.forward(dp_2.y)

# -- 逆伝播 --
def bp(t):
    ol.backward(t)
    dp_2.backward(ol.grad_x)
    ml_2.backward(dp_2.grad_x)
    dp_1.backward(ml_2.grad_x)
    ml_1.backward(dp_1.grad_x)

# -- 重みとバイアスの更新 --
def uppdate_wb():
    ml_1.update(eta)
    ml_2.update(eta)
    ol.update(eta)

# -- 誤差を計算 --
def get_error(t, batch_size):
    return -np.sum(t * np.log(ol.y + 1e-7)) / batch_size  # 交差エントロピー誤差

# csvを読んで要素をintにする
def str2int(path):
    with open(path) as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader]).astype(np.int64)
    return data

def save2csv(file_path, mode, data):
    with open(file_path, mode) as csvfile:
        writer = csv.writer(csvfile)
        if len(np.shape(data)) == 2: # 2次元配列のとき
            writer.writerows(data)
        else:
            writer.writerow(data)

def s(_data):
    print(type(_data),np.shape(_data))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default="",
                        help='directory of the experiment')
    args = parser.parse_args()

    # ==========データの読み込み===========
    load_data = LoadData(args.dir)
    load_data.load()
    target = torch.tensor(load_data.target)
    data = load_data.data
    one_hot = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=load_data.cluster_num).numpy()
    input_train, input_valid, correct_train, correct_valid = train_test_split(data, one_hot, shuffle=True)
    n_train = input_train.shape[0] #訓練用データのサンプル数
    n_test = input_valid.shape[0]    #テスト用データのサンプル数
    _DIR_HERE =  os.path.dirname(os.path.abspath(__file__))
    #==================================

    #=============NNの定義==============
    n_in = np.shape(input_train)[1]
    n_mid = n_in * 4
    n_out = load_data.cluster_num
    wb_width = 0.1
    eta = 0.01

    epoch = 200
    batch_size = 100
    interval = 50

    dic_nn = {
        'n_in':n_in,
        'n_mid':n_mid,
        'n_out':n_out,
        'wb_width':wb_width,
        'eta':eta
    }
    with open('NN_model/model.json', 'w') as f:
        json.dump(dic_nn, f, indent=2, ensure_ascii=False)
    #==================================

    # -- 各層の初期化 --
    ml_1 = MiddleLayer(n_in, n_mid)
    dp_1 = Dropout(0.5)
    ml_2 = MiddleLayer(n_mid, n_mid)
    dp_2 = Dropout(0.5)
    ol = OutputLayer(n_mid, n_out)
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
        fp(input_valid, False)
        error_test = get_error(correct_valid, n_test)
        
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
    plt.savefig("NN_model/Error.png")
    # plt.show()


    #==========学習した重みを保存==============
    # dic_wb = {
    #     'w_ml_1':ml_1.w,
    #     'b_ml_1':ml_1.b,
    #     'w_ml_2':ml_2.w,
    #     'b_ml_2':ml_2.b,
    #     'w_ol':ol.w,
    #     'b_ol':ol.b
    # }
    # with open('NN_wb.json', 'w') as f:
    #     json.dump(dic_wb, f, indent=2, ensure_ascii=False)


    w_ml_1 = _DIR_HERE + '/NN_model/wb/w_ml_1.csv'
    b_ml_1 = _DIR_HERE + '/NN_model/wb/b_ml_1.csv'
    w_ml_2 = _DIR_HERE + '/NN_model/wb/w_ml_2.csv'
    b_ml_2 = _DIR_HERE + '/NN_model/wb/b_ml_2.csv'
    w_ol = _DIR_HERE + '/NN_model/wb/w_ol.csv'
    b_ol = _DIR_HERE + '/NN_model/wb/b_ol.csv'
    # print(ml_1.w.tolist()[:3])
    save2csv(w_ml_1, 'w', ml_1.w.tolist())
    save2csv(b_ml_1, 'w', ml_1.b.tolist())
    save2csv(w_ml_2, 'w', ml_2.w.tolist())
    save2csv(b_ml_2, 'w', ml_2.b.tolist())
    save2csv(w_ol, 'w', ol.w.tolist())
    save2csv(b_ol, 'w', ol.b.tolist())
    #==================================



    #正解率の計算
    fp(input_train,False)
    count_train = np.sum(np.argmax(ol.y,axis=1) == np.argmax(correct_train, axis=1))

    fp(input_valid,False)
    count_test = np.sum(np.argmax(ol.y,axis=1) == np.argmax(correct_valid, axis=1))

    print('学習データでの正解数---{}/{}'.format(count_train,n_train))
    print('テストデータでの正解数---{}/{}'.format(count_test,n_test))

    print("Accuracy Train:" + str(float(count_train)/n_train*100) + "%",
        "Accuracy Test:" + str(float(count_test)/n_test*100) + "%")