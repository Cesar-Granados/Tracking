#!/usr/bin/env python
import numpy as np
class orbslam_point:

	def __init__(self, x, y, z):
	        self.pt = -1#[x,y,z]
	        self.v = []
		self.r = []
		self.t = []
	        self.centroid = -1#[cx,cy,cz]
	        self.pose_camera = -1#[pcx, pcy, pcz]
	        self.camera_distance = -1#Magnitud
		self.set_point3d(x, y, z)

    
	def set_point3d(self, x, y, z):
		self.pt = [x,y,z]
	        return 1


	def points2vector(self):
    		x = self.pose_camera[0] - self.pt[0]
    		y = self.pose_camera[1] - self.pt[1]
    		z = self.pose_camera[2] - self.pt[2]
    		self.v = [x,y,z]
    		return 1
	

	def ec_parametrica(self,t):
	    	x = float(self.pt[0]) + (float(self.pose_camera[0])*t)
    		y = float(self.pt[1]) + (float(self.pose_camera[1])*t)
    		z = float(self.pt[2]) + (float(self.pose_camera[2])*t)
    		self.r = [x,y,z]
    		return 1


	def val_t(self):
    		tx = float(self.pt[0])/float(self.pose_camera[0])# * -1
    		ty = float(self.pt[1])/float(self.pose_camera[1])# * -1
    		tz = float(self.pt[2])/float(self.pose_camera[2])# * -1
    		self.t = [tx,ty,tz]
    		return 1
    
    
	def set_camera_pose(self, x, y, z):
        	self.pose_camera = [x,y,z]
        	return 1
    

	def set_distance(self):
        	'''self.camera_distance = np.sqrt(
        	np.power(self.pose_camera[0] - self.pt[0],2) + 
        	np.power(self.pose_camera[1] - self.pt[1],2) + 
        	np.power(self.pose_camera[2] - self.pt[2],2) )'''
		self.camera_distance = np.sqrt(
        	np.power(0 - self.pt[0],2) + 
        	np.power(0 - self.pt[1],2) + 
        	np.power(0 - self.pt[2],2) )
        	return 1
