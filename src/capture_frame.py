#!/usr/bin/env python3
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from rostopic import get_topic_type
from cv_bridge import CvBridge
import rospy
import cv2
import time
import sys


def image_callback(data):
    print(data.header)
    if compressed_input:
        im = bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
    else:
        im = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    print("tmp/{}.png".format(time.time_ns()))
    cv2.imwrite("tmp/{}.png".format(time.time_ns()), im)
    image_sub.unregister()
    rospy.spin()


rospy.init_node("snapshot", anonymous=True)
bridge = CvBridge()

topic = sys.argv[1] if len(sys.argv) > 1 else "/cam_1/color/image_raw"
print(topic)
input_image_type, input_image_topic, _ = get_topic_type(topic, blocking=True)
print(input_image_type, input_image_topic, _)

compressed_input = input_image_type == "sensor_msgs/CompressedImage"

if compressed_input:
    image_sub = rospy.Subscriber(input_image_topic, CompressedImage, image_callback, queue_size=1)
else:
    image_sub = rospy.Subscriber(input_image_topic, Image, image_callback, queue_size=1)
rospy.spin()
