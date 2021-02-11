# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
import csv
import math
import os
# import NeuralNetwork


#---------------------------データの前処理------------------------------
class LoadData:
    def __init__(self):
        self.dir_here = os.path.dirname(os.path.abspath(__file__))
        self.data = None
        self.target = None
        self.n_in = None
        self.cluster_num = None

    def str2float(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array([row for row in reader]).astype(np.int64)
        return data

    def load(self):
        # 正解データを作成
        path = self.dir_here + '/openpose_data_csv/correct_data.csv'
        data = self.str2float(path)[0]
        # ノイズデータの除去
        noize_indexes = np.array([index for index,num in enumerate(data) if num==-1 ])
        self.target = np.delete(data, noize_indexes)

        # 学習データの作成
        path = self.dir_here + '/openpose_data_csv/pose.csv'
        data = self.str2float(path)
        self.data = np.delete(data, noize_indexes, 0)

        n_data = len(self.target)
        self.n_in = np.shape(self.data)[1]
        self.cluster_num = len(set(self.target))


        print('データ数：' + str(n_data))
        print('クラスタ数：' + str(self.cluster_num))
        print('data：' + str(np.shape(self.data)))
        print('target：' + str(np.shape(self.target)))
        print('--------------データを読み込みました-----------------')
#---------------------------------------------------------------------------------