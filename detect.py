# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
from load_data import LoadData
import torch
from sklearn.model_selection import train_test_split
import json
import rospy
import time
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



if __name__ == '__main__':
    rospy.init_node('NN')
    #=============NNの定義==============
    with open('NN_model/model.json') as f:
        df = json.load(f)
    n_in = df['n_in']
    n_mid = df['n_mid']
    n_out = df['n_out']
    wb_width = df['wb_width']
    eta = df['eta']

    # -- 各層の初期化 --
    ml_1 = MiddleLayer(n_in, n_mid)
    # dp_1 = Dropout(0.5)
    ml_2 = MiddleLayer(n_mid, n_mid)
    # dp_2 = Dropout(0.5)
    ol = OutputLayer(n_mid, n_out)

    # -- 順伝播 --
    def fp(x, is_train):
        ml_1.forward(x)
        # dp_1.forward(ml_1.y, is_train)
        # ml_2.forward(dp_1.y)
        # dp_2.forward(ml_2.y, is_train)
        # ol.forward(dp_2.y)
        ml_2.forward(ml_1.y)
        ol.forward(ml_2.y)
    #==================================


    #===========学習済みの重みを読み込む============
    # csvを読んで要素をfloatにする
    def str2float(path):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array([row for row in reader]).astype(np.float64)
        return data
    base_dir = 'NN_model/wb/'
    ml_1.b = str2float(base_dir + 'b_ml_1.csv')
    ml_1.w = str2float(base_dir + 'w_ml_1.csv')
    ml_2.w = str2float(base_dir + 'w_ml_2.csv')
    ml_2.b = str2float(base_dir + 'b_ml_2.csv')
    ol.w = str2float(base_dir + 'w_ol.csv')
    ol.b = str2float(base_dir + 'b_ol.csv')
    
    load_data = LoadData()
    load_data.load()
    data = load_data.data
    # while not rospy.is_shutdown():
    for input_data in data:
        fp(input_data, is_train=False)
        # print(ol.y[0])
        time.sleep(1)
        # index_result = np.argmax(ol.y,axis=1)
        probability = ol.y[0]*100
        print(probability)