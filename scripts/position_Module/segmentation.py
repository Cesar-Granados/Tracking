#!/usr/bin/env python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pybrain.tools.xml.networkreader import NetworkReader
from cluster import cluster
from improve_image import improve_image
class segmentation:

	def __init__(self):
		self.meanshift = None
		self.list_cluster = None
		self.knn = None
		self.trainData = None
		self.answers = None
		self.rnn = None

	
	def init_meanshift(self, kp):
		bandwidth = estimate_bandwidth(kp, quantile=0.1, n_samples=650)
		ms = MeanShift(bandwidth = bandwidth, bin_seeding=True, min_bin_freq=2, cluster_all=False)
		ms.fit(kp)
		self.meanshift = ms
		return

		
	def indice(self, lista, value):
		return [i for i,x in enumerate(lista) if x==value]


	def create_cluster(self, label, n_cluster, kp, des, centro, img):
	    	position = self.indice(label,n_cluster)
	    	size = len(position)
	    	roi_kp, roi_des = [], []

	    	for i in position:
			point = kp[i]
			roi_kp.append(point)
			roi_des.append(des[i])

	    	roi_kp.sort(key = lambda x:x[0],reverse=True)
	    	maxx = roi_kp[0][0]
	    	minx = roi_kp[size-1][0]
	    	roi_kp.sort(key = lambda x:x[1],reverse=True)

	    	maxy = roi_kp[0][1]
	    	miny = roi_kp[size-1][1]
	    	w, h = int(maxx - minx), int(maxy - miny)
	   	top, left = centro[1]+int((h/2)), centro[0]-int((w/2))
	   	bottom, right = centro[1]-int((h/2)), centro[0]+int((w/2))

	    	image = img[bottom:top, left:right]

		obj_cluster = cluster()
		
		obj_cluster.roi_image = image
		obj_cluster.roi_keypoints = roi_kp
		obj_cluster.roi_descriptors = roi_des
		obj_cluster.size = size

		obj_cluster.max_x = maxx
		obj_cluster.min_x = minx
		obj_cluster.max_y = maxy
		obj_cluster.min_y = miny
		obj_cluster.width = w
		obj_cluster.hight = h
		obj_cluster.center = centro
		
	    	return obj_cluster	


	def separate_clusters(self, kp, des, image):
		list_c = []

		for k, i in enumerate(self.meanshift.cluster_centers_,0):
			centro = [int(i[0]), int(i[1])]
			labels = self.meanshift.labels_
			obj_cluster = self.create_cluster(labels.tolist(), k, kp, des, centro, image)

			if obj_cluster.size < 50 or obj_cluster.width <= 15 or obj_cluster.hight <= 15:
				continue
			list_c.append(obj_cluster)

		self.list_cluster = list_c
		return


	def init_knn(self):
		self.knn = cv2.ml.KNearest_create()
		return


	def read_datafile(self, path):
		arrayMaster = []
    		with open(path) as openfileobject:
        		for line in openfileobject:
            			if(".jpg" in line):
                			arrayCLuster = []
                			continue;
            			if("[" in line):
                			numberArray =[]
                			number = ""
                			numberInt = 0;
                			for vector in line:
                    				if(vector == '['):
                        				continue;
                    				if(vector.isdigit()==True):
                        				number=number+vector;
                    				if(vector ==' '):
                        				if(number!=""):
                            					numberInt = int(number)
                            					numberArray.append(numberInt);
                            					number=""
                    				if(vector==']'):
                        				if(number!=""):
                            					numberInt = int(number)
                            					numberArray.append(numberInt);
                            					number=""
                			arrayCLuster.append(numberArray)
            			#if("|--- image end ---|" in line):
            			if("fin-cluster" in line):
                			arrayCLuster.append(numberArray)
                			arrayMaster.append(arrayCLuster)
    		return arrayMaster
	

	def assign_value(self, data, answ,fileData, value):
		for i in fileData:
        		for j in i:
            			data.append(j)
            			answ.append(value)
		return data, answ
		

	def create_data(self, file_bottle, file_notBottle):
		file1 = read_datafile(file_bottle)
    		file2 = read_datafile(file_notBottle)	
    		self.trainData = []
    		self.answers = []

    		self.trainData, self.answers = assign_value(self.trainData, self.answers, file1, 0)
            	self.trainData, self.answers = assign_value(self.trainData, self.answers, file2, 1)

    		self.trainData = np.array(self.trainData).astype(np.float32)
    		self.answers = np.array(self.answers).astype(np.float32)
    		return

	
	def reduction_descriptor(self):
		size = len(descriptores)
    		ratio = int(size/50)
    		new_descriptores = []
    		w = 0
    		for i in range(50):
        		new_descriptores.append(descriptores[w])
        		w = w + ratio
    		return new_descriptores


	def resume_knn(resultados):
    		no_botella = 0
    		for i in resultados:
        		if i == 1:
            			no_botella = no_botella+1
    		if no_botella >= 26:
        		return 1
	    	return 0

#--- Actualizar funcion ---#
	def active_knn(self):
		resultados, botella = [], []
    		for i in self.list_cluster:
        		descriptor = reduction_descriptor(i[9])
        		newcomer = np.array(descriptor).astype(np.float32)
        		ret, results, neighbours, dist = self.knn.findNearest(newcomer, 50)
        		resultados.append(resume_knn(results))

    		for i,j in enumerate(resultados):
        		if j == 0:
            			botella.append(cluster[i])
		return botella
	

	def init_rnn(self, path):
		self.rnn = NetworkReader.readFrom(path)
		return
	
	
	def active_rnn(self):
		improve = improve_image()
		botella = []
		for i in self.list_cluster:
        		hu = improve.Hu(i.roi_image)
			respuesta,no_datos = 0, 0
			for k,j in enumerate(i.roi_descriptors):
				data = np.append(hu,j)
        			respuesta = respuesta + self.rnn.activate(data)
				no_datos = k
			respuesta = respuesta/no_datos
        		if respuesta < 0.6:
            			botella.append(i)
		return botella
