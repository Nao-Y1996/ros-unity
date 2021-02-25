#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import csv
import math
import os
from load_data import LoadData
# import torch
# from sklearn.model_selection import train_test_split
import json
import time
from std_msgs.msg import Float32MultiArray

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


if __name__ == '__main__':
    dir_here =  os.path.dirname(os.path.abspath(__file__))
    rospy.init_node('detection')
    #=============NNの定義==============
    with open(dir_here + '/NN_model/model.json') as f:
        df = json.load(f)
    n_in = df['n_in']
    n_mid = df['n_mid']
    n_out = df['n_out']
    wb_width = df['wb_width']
    eta = df['eta']

    # -- 各層の初期化 --
    ml_1 = MiddleLayer(n_in, n_mid)
    ml_2 = MiddleLayer(n_mid, n_mid)
    ol = OutputLayer(n_mid, n_out)

    # -- 順伝播 --
    def fp(x, is_train):
        ml_1.forward(x)
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
    base_dir = dir_here + '/NN_model/wb/'
    ml_1.b = str2float(base_dir + 'b_ml_1.csv')
    ml_1.w = str2float(base_dir + 'w_ml_1.csv')
    ml_2.w = str2float(base_dir + 'w_ml_2.csv')
    ml_2.b = str2float(base_dir + 'b_ml_2.csv')
    ol.w = str2float(base_dir + 'w_ol.csv')
    ol.b = str2float(base_dir + 'b_ol.csv')
    #============================================


    situation_pub = rospy.Publisher('situation', Float32MultiArray, queue_size=10)
    situataions = Float32MultiArray()
    def detect(message):
        pose_now = np.delete(message.data, [2, 5, 8, 11, 14, 17, 20,
                        23, 26, 29, 32, 35, 38, 41, 44])
        fp(pose_now, is_train=False)
        situataions.data = (ol.y[0]*100).tolist()
        situation_pub.publish(situataions)



    while not rospy.is_shutdown():
        
        sub = rospy.Subscriber('/OpenPose', Float32MultiArray, detect)
        time.sleep(1)
        # situation_pub.publish(param.situataions)
