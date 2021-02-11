#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
import os

def save2csv(file_path, mode, data):
    with open(file_path, mode) as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow(data)

class SituationDetector():
    def __init__(self):
        self.situation_pub = rospy.Publisher('situation', String, queue_size=10)
        self.joint_num = 15
        self.pose_old = np.zeros(self.joint_num * 2)
        self.joint_now = np.zeros((self.joint_num, 2))
        self.joint_old = np.zeros((self.joint_num, 2))
        self.joint_diff = np.zeros(self.joint_num)

    def detect(self, message):
        # 信頼値の部分を削除
        pose_now = np.delete(message.data, [2, 5, 8, 11, 14, 17, 20,
                        23, 26, 29, 32, 35, 38, 41, 44])
        # 現在時刻のOpenPoseの骨格情報
        pose_now = np.array(pose_now, dtype=np.int64)
        save2csv('openpose_data_csv/pose.csv', mode='a', pose_now)
        
        # 各jointにおける、1時刻前との距離 joint_diff
        self.joint_now = np.reshape(pose_now, (self.joint_num,2))
        for i in range(self.joint_num):
            self.joint_diff[i] = np.linalg.norm(self.joint_now[i] - self.joint_old[i])
        save2csv('openpose_data_csv/joint_diff.csv', mode='a', self.joint_diff)

        # ユークリッド距離
        EuclideanDistance =np.linalg.norm(pose_now - self.pose_old)
        save2csv('openpose_data_csv/EuclideanDistance.csv', mode='a', [EuclideanDistance])

        self.joint_old = self.joint_now
        self.pose_old = pose_now


        # ここに状況判断の処理

        msg = String()
        msg.data = 'situation: %s'%rospy.get_time()
        self.situation_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('sub_pose')
    situation_detector = SituationDetector()
    dir_here =  os.path.dirname(os.path.abspath(__file__))

    f = open(dir_here + '/openpose_data_csv/diff.csv', 'w')
    f.close
    f = open(dir_here + '/csopenpose_data_csvv/pose.csv', 'w')
    f.close
    f = open(dir_here + '/openpose_data_csv/EuclideanDistance.csv', 'w')
    f.close
    while not rospy.is_shutdown():

        sub = rospy.Subscriber('/OpenPose', Float32MultiArray, situation_detector.detect)
        rospy.spin()



