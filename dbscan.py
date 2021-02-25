#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ライブラリのインポート
import rospy
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
# from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph


class DBSCAN():
    def __init__(self, data, eps, min_samples):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_data = None
        self.cluster_num = None
        self.noise_num = None
        self.calc_time = None

    def calc(self):
        self.dbscan_data = cluster.DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit_predict(self.data)
        self.noise_num = sum(self.dbscan_data == -1)
        if self.noise_num == 0:
            self.cluster_num = len(set(self.dbscan_data))
        else:
            self.cluster_num = len(set(self.dbscan_data)) - 1


def cluster_plots(data, colors='gray', title1='Dataset 1'):  # グラフ作成
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title1, fontsize=14)
    body_part = 1
    body_part_index = body_part * 2
    ax1.scatter(data[:, body_part_index], data[:, body_part_index+1], s=8, lw=0, c=colors)
    # plt.xlim(0,1920)
    # plt.ylim(0,1080)
    fig.savefig('webcamera_pose_data/dbscan_result.png')

# -----------OpoenPoseデータの読み込み-----------
with open('webcamera_pose_data/pose_par_second.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
data = np.array([[int(float(v)) for v in row] for row in l])

# 信頼値の削除
max_body_parts = 18
confidence_score_indexes = list(range(2, 3*max_body_parts, 3))
data = np.delete(data, confidence_score_indexes, 1)
                         
print('データ形状：' + str(np.shape(data)) )
dbscan = DBSCAN(data, eps=500, min_samples=120)
dbscan.calc()
# print(np.reshape(dbscan.dbscan_data, (-1,1)))
with open('webcamera_pose_data/correct_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(dbscan.dbscan_data)
print("クラスタ：" + str(dbscan.cluster_num) +  "ノイズ：" + str(dbscan.noise_num) )

cluster_plots(data, dbscan.dbscan_data)