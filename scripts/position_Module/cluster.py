#!/usr/bin/env python
class cluster:
	def __init__(self):
		self.roi_image = None
		self.roi_keypoints = None
		self.roi_descriptors = None
		self.size = None

		self.min_x = None
		self.max_x = None
		self.min_y = None
		self.max_y = None
		self.width = None
		self.hight = None
		self.center = None
		
		self.histogram_hsv = None
		self.track_image = None
		self.track_window = None
		self.track_ant = None


	def init_track_window(self):
		self.track_window = (self.center[0], self.center[1], self.width, self.hight)
