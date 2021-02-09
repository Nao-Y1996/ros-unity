#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

rospy.init_node('pub')
pub = rospy.Publisher('chatter', String, queue_size=10)
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    msg = String()
    msg.data = 'hello %s'%rospy.get_time()
    pub.publish(msg)
    rate.sleep()
# if __name__ == '__main__':

#     face_tracking()
