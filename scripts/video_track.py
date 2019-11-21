#!/usr/bin/env python
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import math
import cv2
import math
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from custom_msg2.msg import custom_msg
import sensor_msgs.point_cloud2 as pc2
#|---------------|#
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
#|---------------------------------------------------|#
#|------|#
import time as t
#|--------------------------------------------------- Global -------------------------------------------------|#
#|--- Posicion de la camara ---|#
pose_camara = 0
point_cloud_all = 0
keypoint_cloud = 0
track_point = 0
#|--------------------------------------------------- ROS ----------------------------------------------------|#
def listener():
	#|--- Organizacion ---|#
	#|--- Descargar Base de datos ---|#
	trainData, responses = conversion_matriz()
	#|--- Entrenar KNN ---|#
	knn = entrenar_knn(trainData, responses)
	#|--- Inicializar variables de seguimiento ---|#
	track_window, roi_hist, term_crit = [0,0,0,0],0,( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	#|--- Variables de control ---|#
	control, cluster = True, []
	#|--- Iniciar nodo ---|#
	rospy.init_node('Imagen', anonymous=True)
	try:	
		rospy.Subscriber("/camera_pose", PoseStamped, callback, queue_size=1)
		rospy.Subscriber("/point_cloud_ref", PointCloud2, callback2, queue_size=1)
		rospy.Subscriber("/KeyPoints", custom_msg, callback3, queue_size=1)
		rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, callback4, [knn, track_window, roi_hist, term_crit, control, cluster, 0], queue_size=1)

	except (rospy.ROSException),e:
		print(e)
	rospy.spin()

def callback(data):
	global pose_camara
	pose_camara = data.pose#| Estructura: (x, y, z) |#
	#print(pose_camera)


def callback2(data):
	#|--- variables locales ---|#
	global point_cloud_all
	list_points = []
	points = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
	#|--- crear lista ---|#
	for i in points:
		list_points.append(i)
	#|--- asignar a la variable global ---|#
	point_cloud_all = list_points#| Estructura: (x, y, z) |#
	#print("|--- pc ---|")


def callback3(data):
	#|--- variables locales/globales ---|#
	global keypoint_cloud
	list_keypoints = []
	#|--- crear lista ---|#
	for i in data.points:
		list_keypoints.append(i)
	#|--- asginar a la variable global ---|#
	keypoint_cloud = list_keypoints#| Estructura : descriptor de 7 elementos |#
	#print("|--- kp ---|")


def callback4(data, args):
	global pose_camara
	global point_cloud_all
	global keypoint_cloud
	#|------------------------------|#
	#tiempo = t.time()
	cloud = []
	if pose_camara != None:
		for i in point_cloud_all:
			orb_point = orbslam_point(i[0], i[1], i[2])
			orb_point.set_camera_pose(pose_camara.position.x, pose_camara.position.y, pose_camara.position.z)
			orb_point.set_distance()#Distance to origin
			orb_point.points2vector()
			orb_point.val_t()
			orb_point.ec_parametrica(orb_point.t[2])
			cloud.append(orb_point)
		cloud.sort(key = lambda x: x.camera_distance, reverse = False)
	#|------------------------------|#
	#|--- Extraer imagen comprimida ---|#
	np_arr = np.fromstring(data.data, np.uint8)
	#|--- Codificar en formato opencv() ---|#
	img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	#|--- imagen ---|#
	if img_np is not None:
		#|--- Main ---|#
		centroide, args[1], args[2], args[4], args[5], imagen  = main(img_np, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		#|--------------------------------------|#
		kp = centroide
		#|--- Grupo es una lista de objetos orbslam_point() ---|
		grupo, c = [], 3
		while grupo == [] :
			for i in cloud:
				grupo = agrupar(kp,i, grupo, c)
			c = c + 3
		#tiempo = t.time() - tiempo
		#|--- Publicar resultado de analisis ---|#'''
		org0 = (0,445)		
		org1 = (0,460)
		org2 = (0,475)
		font = cv2.FONT_HERSHEY_PLAIN#cv2.FONT_HERSHEY_TRIPLEX
		fontscale = 1
		color = (0,255,255)
		thikness = 1
		down = False
		#|--------------------------------------|#
		centro = pt_centro(grupo)
		dist = distancia((pose_camara.position.x,pose_camara.position.y,pose_camara.position.z), centro)
		string0 = "Origen: "+"("+str(pose_camara.position.x)+","+str(pose_camara.position.y)+","+str(pose_camara.position.z)+")"
		string1 = "Destino: "+"("+str(centro[0])+","+str(centro[1])+","+str(centro[2])+")"
		string2 = "Distancia: "+str(dist)
		cv2.putText(imagen, string0, org0, font, fontscale, color, thikness, cv2.LINE_AA, down)
		cv2.putText(imagen, string1, org1, font, fontscale, color, thikness, cv2.LINE_AA, down)
		#print string1
		cv2.putText(imagen, string2, org2, font, fontscale, color, thikness, cv2.LINE_AA, down)
		cv2.imshow("Imagen ORB", imagen)
		cv2.waitKey(1)
		talker(centro)


def talker(posicion):
	#|--- Generar publisher ---|#
	pub = rospy.Publisher('Distancia', Float32, queue_size=1)
	#rate = rospy.Rate(1) # 10hz
	if not rospy.is_shutdown():
		#|--- Publicar resultado ---|#
		pub.publish(posicion)
		#print posicion
		#rate.sleep()
#|--------------------------------------------------- Clases ----------------------------------------------------------|#
def	distancia(p1,p2):	
	distance = np.sqrt(
        np.power(p2[0] - p1[0],2) + 
        np.power(p2[1] - p1[1],2) +
	np.power(p2[2] - p1[2],2) )
	return distance

def	pt_centro(list):
	promedio = [0,0,0]
	c = len(list)
	for i in list:
		promedio[0] = promedio[0] + i.pt[0]
		promedio[1] = promedio[1] + i.pt[1]
		promedio[2] = promedio[2] + i.pt[2]
	promedio[0] = promedio[0]/c
	promedio[1] = promedio[1]/c
	promedio[2] = promedio[2]/c
	return promedio

def	agrupar(kp, pt, list, c):
	minx,maxx,miny,maxy = (float(kp[0])-c)/1000, (float(kp[0])+c)/1000, (float(kp[1])-c)/1000, (float(kp[1])+c)/1000
	point = [(pt.r[0]),(pt.r[1])]
	if point[0] > minx:
		if point[0] < maxx:
			if point[1] > miny:
				if point[1] < maxy:
					list.append(pt)
	return list

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
    	tx = float(self.pt[0])/float(self.pose_camera[0]) * -1
    	ty = float(self.pt[1])/float(self.pose_camera[1]) * -1
    	tz = float(self.pt[2])/float(self.pose_camera[2]) * -1
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
#|--------------------------------------------------- Organizacion ----------------------------------------------------|#
def r_archivo(path):
    arrayMaster = []
    #|--- Parseo ---|#
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
    #|--- Retornar ---|#
    return arrayMaster


def conversion_matriz():
    #|---- Descargar descriptores ----|#
    file1 = r_archivo("/home/cesar/catkin_ws/src/tracking/src/Descriptores/Botella.txt")
    file2 = r_archivo("/home/cesar/catkin_ws/src/tracking/src/Descriptores/NoBotella.txt")
    #|--- variables locales ---|#	
    trainData = []
    responses = []
    #|---- Organizar informacion ----|#
    for i in file1:
        for j in i:
            trainData.append(j)
            responses.append(0)
            
    for i in file2:
        for j in i:
            trainData.append(j)
            responses.append(1)
    #|---- Establecer formato ----|#
    trainData = np.array(trainData).astype(np.float32)
    responses = np.array(responses).astype(np.float32)
    #|--- Retornar ---|#
    return trainData, responses


def entrenar_knn(trainData,responses):
    #|--- Crear objeto knn ---|#
    knn = cv2.ml.KNearest_create()
    #|--- Entrenar con las respectivas bases de datos ---|#
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    #|--- Retornar ---|#
    return knn


def mejorar_imagen(img):
    #img = convol(img, 1)
    img = ecualizar(img)
    return img

def ecualizar(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def orb_features(img):
    #|--- Orb features ---|#
    orb = cv2.ORB_create(nfeatures=650, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)    
    kp, des = orb.detectAndCompute(img, None)
    #img2 = cv2.drawKeypoints(img,kp, None,color=(0,255,0), flags=0)
    kp = np.array([kp[idx].pt for idx in range(0, len(kp))]).astype(int)
    return kp, des


def meanshift_cl(kp, des, img):
    #|--- variables locales ---|#
    cluster_final = []
    #|--- Meanshif var ---|#
    bandwidth = estimate_bandwidth(kp, quantile=0.1, n_samples=650)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True, min_bin_freq=2, cluster_all=False)
    ms.fit(kp)
    labels = ms.labels_
    lista_label = labels.tolist()
    cluster_center = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    #print("Clusters: ", n_clusters_)
    #|--- Analizar cada cluster del frame ---|#
    for k,i in enumerate(cluster_center,0):
        centro = [int(i[0]), int(i[1])]
	#|--- Obteniendo los valores maximos y minimos del cluster ---|#
        cluster = max_min(lista_label,k, kp, des, centro, img.copy())#|Problema al igualar en nullo.|#
	#|--- Eliminar cluster que no aportan informacion al sistema ---|#
        if cluster[7] < 50 or cluster[4] <= 10 or cluster[5] <= 10:
            continue
	#|--- Almacenar cluster que cumplen con los objetivos ---|#
        cluster_final.append(cluster)
    #|--- Retornar ---|#
    return cluster_final


def max_min(label, cluster, kp, des, centro, img):
    #|--- Obtener la posicion de los elementos del cluster ---|#
    position = indice(label,cluster)
    #|--- Calcular la longitud del cluster|#
    size = len(position)
    #|--- Variables locales ---|#
    roi_kp, roi_des = [], []
    #|--- Buscar puntos maximos y minimos ---|#
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
    #|--- Calculo del ancho y alto del cluster ---|#
    w, h = int(maxx - minx), int(maxy - miny)
    #|--- ROI image ---|#
    top, left = centro[1]+int((h/2)), centro[0]-int((w/2))
    bottom, right = centro[1]-int((h/2)), centro[0]+int((w/2))
    imagen = img[bottom:top, left:right]
    #|--- Organizar caracteristicas ---|#
    cluster = [minx, maxx, miny, maxy, w, h, centro, size, roi_kp, roi_des, imagen]
    #|--- Retornar ---|#
    return cluster

def indice( lista, value):
    return [i for i,x in enumerate(lista) if x==value]


def k_nearest(knn, cluster):
    botella = []
    net = NetworkReader.readFrom('/home/cesar/catkin_ws/src/tracking/src/RedNeuronal.xml')
    for i in cluster:
        imagen = i[10]
        hu = Hu(imagen)
        respuesta = abs(net.activate(hu))
        if respuesta < 0.6:
            botella.append(i)
    '''
    resultados, botella = [], []
    #|--- Organizar cluster ---|#
    for i in cluster:
        #|--- reduccion a 50 descriptores ---|#
        descriptor = reduccion(i[9])
        #|--- estandarizar al knn ---|#
        newcomer = np.array(descriptor).astype(np.float32)
        #|--- aplicar knn ---|#
        ret, results, neighbours, dist = knn.findNearest(newcomer, 7)
        #|--- Almacenar resultados ---|#
        resultados.append(resultado_knn(results))
    #|--- Analisar resultados ---|#
    for i,j in enumerate(resultados):
        if j == 0:
            botella.append(cluster[i])
    #|--- Retornar ---|#'''
    return botella

#|----------------------------------------------------------------------------|#
def calculo(imagen):
    imagen2 = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    #imagen2 = ecualizar(imagen)
    histograma = cv2.calcHist([imagen2],[0],None,[180],[0,179])
    histograma = cv2.normalize(histograma,  None, 0, 180, cv2.NORM_MINMAX)
    colores = mediacolores(histograma)
    mask = mascara(imagen, colores)
    return mask, colores

def tlog(hu):
    if hu[0] > 0:
    	#print hu[0]
    	hu[0] = -np.sign(hu[0]) * np.log10(np.abs(hu[0]))
    if hu[1] > 0:
	#print hu[1]
	hu[1] = -np.sign(hu[1]) * np.log10(np.abs(hu[1]))
    if hu[2] > 0:
    	#print hu[2]
    	hu[2] = -np.sign(hu[2]) * np.log10(np.abs(hu[2]))
    if hu[3] > 0:
    	#print hu[3]
    	hu[3] = -np.sign(hu[3]) * np.log10(np.abs(hu[3]))
    if hu[4] > 0:
    	#print hu[4]
    	hu[4] = -np.sign(hu[4]) * np.log10(np.abs(hu[4]))
    if hu[5] > 0:
    	#print hu[5]
    	hu[5] = -np.sign(hu[5]) * np.log10(np.abs(hu[5]))
    if hu[6] > 0:
    	#print hu[6]
    	hu[6] = -np.sign(hu[6]) * np.log10(np.abs(hu[6]))
    return hu

def Hu(imagen):
    mask, color = calculo(imagen)
    momentos = cv2.moments(mask)
    humoments = cv2.HuMoments(momentos)
    humoments = tlog(humoments)
    humoments = np.vstack((humoments,color[0]))
    humoments = humoments.flatten()
    return humoments
#|----------------------------------------------------------------------------|#

def reduccion(descriptores):
    size = len(descriptores)
    ratio = int(size/50)
    new_descriptores = []
    w = 0
    for i in range(50):
        new_descriptores.append(descriptores[w])
        w = w + ratio
    return new_descriptores


def resultado_knn(resultados):
    no_botella = 0
    #|--- Comprobar resultados ---|#
    for i in resultados:
        if i == 1:
            no_botella = no_botella+1
    #|--- Verificar respuestas ---|#
    if no_botella >= 26:
        return 1
    return 0

def preparar_track(list_clusters):
    #|--- Discriminar por el de mayor informacion ---|#
    list_clusters = sort(list_clusters)
    cluster = list_clusters[0]
    roi_img = cluster[10]
    hsv_roi = roi_img
    cv2.imshow("Coca", hsv_roi)
    hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Coca", hsv_roi)
    #|--- Make a point window ---|#
    centro = cluster[6]
    w, h = cluster[4], cluster[5]
    x, y, w, h = centro[0], centro[1], w, h
    track_window = (x, y, w, h)
    #|--- Make a mask ---|#
    histograma = cv2.calcHist([hsv_roi],[0],None,[180],[0,179])
    histograma = cv2.normalize(histograma,  None, 0, 180, cv2.NORM_MINMAX)
    colores = mediacolores(histograma)
    mask = mascara(hsv_roi.copy(),colores)
    #cv2.imshow("Mascara", mask)
    roi_hist = cv2.calcHist([hsv_roi], [0,1], mask, [180,256], [0,179,0,255])
    cv2.normalize(roi_hist, roi_hist, 0, 180, cv2.NORM_MINMAX)
    #|--- Retornar ---|#
    return track_window, roi_hist, cluster

def sort(cluster):
    #|--- Ordenamiento por tamano de cluster ---|#
    cluster.sort(key = lambda x: x[7], reverse = True)
    #|--- Retornar ---|#
    return cluster

def mediacolores(imagen):
    rojo, naranja, amarillo, verde = [],[],[],[]
    azul, morado, rosa = [],[],[]
    #|--------------------------|#
    rojo = np.vstack((imagen[0:10],imagen[169:179]))
    naranja = imagen[11:25]
    amarillo = imagen[26:35]
    verde = imagen[36:70]
    azul = imagen[71:129]
    morado = imagen[130:139]
    rosa = imagen[140:168]
    #|--------------------------|#
    colors = []
    colors.append(promedio(rojo,0))
    colors.append(promedio(naranja,1))
    colors.append(promedio(amarillo,2))
    colors.append(promedio(verde,3))
    colors.append(promedio(azul,4))
    colors.append(promedio(morado,5))
    colors.append(promedio(rosa,6))
    colors.sort(key= lambda x:x[0][0], reverse=True)
    #|--------------------------|#
    return colors[0]

def promedio(list,n):
    k = 0
    for i in list:
        k = k + i
    k = k/len(list)
    return [k,n]

def mascara(imagen,colores):
    colores[1] = 0
    print colores
    #|--- Mascara de rojos hsv ---|#
    if colores[1] == 0:
        low, upp = (170,100,100),(180,255,255)
        low2, upp2 = (0,100,100),(10,255,255)
        imagen1 = cv2.inRange(imagen,low, upp)
        imagen2 = cv2.inRange(imagen,low2, upp2)
        #print "rojo"
        return cv2.add(imagen1,imagen2)
    
    if colores[1] == 1:
        low, upp = (11,100,100),(25,255,255)
        #print "naranja"
        return cv2.inRange(imagen,low, upp)
    
    if colores[1] == 2:
        low, upp = (26,100,100),(35,255,255)
        #print "amarillo"
        return cv2.inRange(imagen,low, upp)
    
    if colores[1] == 3:
        low, upp = (36,100,100),(70,255,255)
        #print "verde"
        return cv2.inRange(imagen,low, upp)
    
    if colores[1] == 4:
        low, upp = (71,100,100),(129,255,255)
        #print "azul"
        return cv2.inRange(imagen,low, upp)
    
    if colores[1] == 5:
        low, upp = (130,100,100),(139,255,255)
        #print "morado"
        return cv2.inRange(imagen,low, upp)
    
    if colores[1] == 6:
        low, upp = (140,100,100),(168,255,255)
        #print "rosa"
        return cv2.inRange(imagen,low, upp)
    #|--- Retornar ---|#


def seguimiento(frame, roi_hist, track_window, term_crit ):
    global track_point
    #|--- Convertir imagen a formato HSV ---|#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #|--- Calculate backprojection ---|#
    dst = cv2.calcBackProject([frame],[0,1],roi_hist,[0,179,0,255],1)
    cv2.imshow("hist", dst)
    #|--- Calculate a new location ---|#
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #|--- Draw a new region ---|#
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    if track_point == 0:
    	x, y, w, h = track_window
	x, y, w, h = x-1, y-1, w-1, h-1
	track_point = track_point+1
    else:
	x, y, w, h = track_window
	x, y, w, h = x+1, y+1, w+1, h+1
	track_point = 0
    x1, y1, x2, y2 = (x-int(w/2)), (y+int(h/2)), (x+int(w/2)), (y-int(h/2)) 
    frame2 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    #|--- Retornar ---|#
    return frame2, track_window

def perdida_objeto(track, track_ant):
	x1, y1, w1, h1 = track
	x2, y2, w2, h2 = track_ant
	const_despl = 30
	#--- Calcular desplazamiento ---#
	eje_x, eje_y = x1 - x2, y1 - y2
	if eje_x < 0: eje_x = eje_x * -1
	if eje_y < 0: eje_y = eje_y * -1
	if eje_x > const_despl or eje_y > const_despl:
		return True
	return False

#|--------------------------------------------------- Main ---------------------------------------------------|#
def main(imagen, knn, track_window, roi_hist, term_crit, control, cluster, posicion_camera):
	#global crono, promedio
	track_ant = track_window
	#|--- Mejoramiento de la imagen ---|#
	img = mejorar_imagen(imagen.copy())
	if control is True:
		#|--- ORB --|#
		kp, des = orb_features(img.copy())
		#|--- Meanshift ---|#
		cluster = meanshift_cl(kp, des, img.copy())
		#|--- KNN ---|#
		result = k_nearest(knn, cluster)
		#|--- Evitar listas vacias ---|#
		if result == None or len(result) == 0:
			#|--- Retornar mismos valores ---|#
			return 0, [0,0,0,0], None, True, []
		#|--- Pre-Track ---|#
		track_window, roi_hist, cluster = preparar_track(result)
		#control = False
	#|--- Tracking ---|#
	track_img, track_window = seguimiento(img.copy(), roi_hist, track_window, term_crit)
	return cluster[6], track_window, roi_hist, control, cluster, track_img
#|--------------------------------------------------- Inicio ---------------------------------------------------|#
if __name__ == '__main__':
	#|--- Ejecutar hilo ---|#
	listener()
    	cv2.destroyAllWindows()
	print("|--- Fin ---|")
