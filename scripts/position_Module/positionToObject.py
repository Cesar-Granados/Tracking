#!/usr/bin/env python
#--- Own modules ---#
from improve_image import improve_image
from segmentation import segmentation
from cluster import cluster
from orbslam_point import orbslam_point
#--- Opencv-others ---#
import cv2
import numpy as np
import math
import time as t
#--- Scikit-learn ---#
from sklearn.cluster import MeanShift, estimate_bandwidth
#--- ROS ---#
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from custom_msg2.msg import custom_msg
#--- Pybrain ---#
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
#--- Variables Globales ---#
count_t = 0
control = True
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
sgt = segmentation()
'''
sgt.init_knn()
Bottle = "/home/cesar/catkin_ws/src/tracking/Utility/Descriptores/Botella.txt"
NotBottle = "/home/cesar/catkin_ws/src/tracking/Utility/Descriptores/NoBotella.txt"
sgt.create_data(Bottle,NotBottle)
'''
Rnn = "/home/cesar/catkin_ws/src/tracking/Utility/RNN/RedNeuronalInt 40-80(0.4-0.95).xml"
sgt.init_rnn(Rnn)

cl = cluster()
#--- Funciones ---#
def distancia(p1,p2):	
	distance = np.sqrt(
        np.power(p2[0] - p1[0],2) + 
        np.power(p2[1] - p1[1],2) +
	np.power(p2[2] - p1[2],2) )
	return distance


def pt_centro(lista):
	lista_camara = []
	for i in lista:
		dst = distancia(i.pose_camera,i.pt)
		lista_camara.append([i,dst])
		
	lista_camara.sort(key = lambda x:x[1], reverse= False)
	promedio = [0,0,0]
	c = 4
	for k,i in enumerate(lista_camara):
		if k < c:
			promedio[0] = promedio[0] + i[0].pt[0]
			promedio[1] = promedio[1] + i[0].pt[1]
			promedio[2] = promedio[2] + i[0].pt[2]
		else:
			break
	promedio[0] = promedio[0]/c
	promedio[1] = promedio[1]/c
	promedio[2] = promedio[2]/c
	return promedio


def	agrupar(kp, pt, list, c):
	minx,maxx,miny,maxy = (float(kp[0])-c), (float(kp[0])+c), (float(kp[1])-c), (float(kp[1])+c)
	point = [(pt.r[0]*1000),(pt.r[1]*1000)]
	if point[0] > minx:
		if point[0] < maxx:
			if point[1] > miny:
				if point[1] < maxy:
					list.append(pt)
	return list


def plain_position(camera_pose, point_cloud, sensor_image):
	global cl
	cloud, centro = [], None
	if camera_pose is not None and sensor_image is not None:
		for i in point_cloud:
			orb_point = orbslam_point(i[0], i[1], i[2])
			orb_point.set_camera_pose(camera_pose.position.x, camera_pose.position.y, camera_pose.position.z)
			orb_point.set_distance()
			orb_point.points2vector()
			orb_point.val_t()
			orb_point.ec_parametrica(orb_point.t[2])
			cloud.append(orb_point)

		cloud.sort(key = lambda x: x.camera_distance, reverse = False)
	
		kp = cl.center
		
		grupo, c = [], 3
		while grupo == [] :
			for i in cloud:
				grupo = agrupar(kp,i, grupo, c)
			c = c + 3
		centro = pt_centro(grupo)
		return centro
	return None


def lost_object(cluster):
	global count_t

	x1, y1, w1, h1 = cluster.track_window
	x2, y2, w2, h2 = cluster.track_ant
	const_despl = 30

	dst = int(math.sqrt((x2-x1)**2+(y2-y1)**2))

	if dst > const_despl:
		count_t = 0
		return True

	count_t = count_t +1
	if count_t >= 5:
		count_t = 0
		return True
	return False


def tracking(cluster, imp_image):
	dst = cv2.calcBackProject([imp_image.HSV_image],[0,1],cluster.histogram_hsv,[0,179,0,255],1)
	
	cluster.track_ant = cluster.track_window
	ret, cluster.track_window = cv2.meanShift(dst, cluster.track_window, ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))
	
	x, y, w, h = cluster.track_window
	x1, y1, x2, y2 = (x), (y), (x+w), (y+h)
	cluster.track_image = cv2.rectangle(imp_image.BGR_image, (x1,y1), (x2,y2), (0,255,0), 2)
	return


def init_track(list_clusters, imp_image):
	list_clusters.sort(key = lambda x:x.size, reverse = True)
	cluster = list_clusters[0]
	
	roi_hsv = cv2.cvtColor(cluster.roi_image, cv2.COLOR_BGR2HSV)
	cluster.init_track_window()
	
	histogram = cv2.calcHist([roi_hsv],[0],None,[180],[0,179])
	histogram = cv2.normalize(histogram,  None, 0, 180, cv2.NORM_MINMAX)

	colors = imp_image.mediacolors(histogram)
	mask = imp_image.mascara(roi_hsv, colors)
	
	cluster.histogram_hsv = cv2.calcHist([roi_hsv], [0,1], mask, [180,256], [0,179,0,255])
	cluster.histogram_hsv = cv2.normalize(cluster.histogram_hsv, None, 0, 180, cv2.NORM_MINMAX)
	return cluster

	
def select_object(camera_pose, point_cloud, sensor_image):
	global imp_image, sgt, cl, control

	if camera_pose is not None and point_cloud is not None and sensor_image is not None:
		imp_image = improve_image()
		imp_image.init_BGR(sensor_image)
		imp_image.init_HSV()
		imp_image.equalize_image()
		
		if control is True:
			imp_image.orb_features()

			sgt.init_meanshift(imp_image.ORB_keypoints)
			sgt.separate_clusters(imp_image.ORB_keypoints, imp_image.ORB_descriptors, imp_image.BGR_image)
			result = sgt.active_rnn()

			if result == None or len(result) == 0:
				control == True

			cl = init_track(result, imp_image)
			control = False
		
		tracking(cl, imp_image)
		control = lost_object(cl)
	return


def image_option(option):
	if option == 1:
		return message_filters.Subscriber("/usb_cam/image_raw/compressed", CompressedImage)
	if option == 2:
		return message_filters.Subscriber("/videofile/image_raw/compressed", CompressedImage)
	if option == 3:
		return message_filters.Subscriber("/frame_now/compressed", CompressedImage)


#--- Callbacks ROS ---#
def callback(Pd, PC2, CI):
		camera_pose = Pd.pose
		
		point_cloud = pc2.read_points(PC2, skip_nans=True, field_names=("x", "y", "z"))

		np_arr = np.fromstring(CI.data, np.uint8)
		sensor_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		
		select_object(camera_pose, point_cloud, sensor_image)

		point = plain_position(camera_pose, point_cloud, sensor_image)
		
		if point is not None:
			talker(rospy, point)


#--- Listeners ROS ---#
def listener():
	try:
		camera_sub = message_filters.Subscriber("/camera_pose", PoseStamped)
		cloud_sub = message_filters.Subscriber("/point_cloud_ref", PointCloud2)
		image_sub = image_option(2)
		ts = message_filters.ApproximateTimeSynchronizer([camera_sub, cloud_sub, image_sub], queue_size=5, slop=1)#4
		ts.registerCallback(callback)
	except (rospy.ROSException), e:
		print(e)


#--- Publishers ROS ---#
def talker(rospy, point):
	try:
		publisher = rospy.Publisher('Object_position', PoseStamped, queue_size=1)
		if not rospy.is_shutdown():
			object_pose = PoseStamped()

			object_pose.header.stamp = rospy.Time.now()
			object_pose.header.frame_id = "ground"

			object_pose.pose.position.x = point[0]
			object_pose.pose.position.y = point[1]
			object_pose.pose.position.z = point[2]

			publisher.publish(object_pose)
			
	except (rospy.ROSException), e:
		print(e)


#--- Main ---#
if __name__ == '__main__':
	rospy.init_node('Object_position', anonymous=True)
	listener()
	cv2.destroyAllWindows()
	rospy.spin()
	
