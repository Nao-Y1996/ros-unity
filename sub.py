#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import os

def callback(message):
    # print(message)
    rospy.loginfo('I got : %s', message.data)

rospy.init_node('sub')
sub = rospy.Subscriber('OpenPose', Float32MultiArray, callback)
rospy.spin()


base_dir =  os.path.dirname(os.path.abspath(__file__))
print(base_dir)

f= open(base_dir + 'diff.csv', 'w')
f.close