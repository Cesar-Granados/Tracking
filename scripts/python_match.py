#!/usr/bin/env python
from sklearn.cluster import MeanShift, estimate_bandwidth
import time as t
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
#|--------------------------------------------------- Global -------------------------------------------------|#
#|--- Posicion de la camara ---|#
pose_camara = 0
point_cloud_all = 0
keypoint_cloud = 0
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
		rospy.Subscriber("/frame_now/compressed", CompressedImage, callback4, [knn, track_window, roi_hist, term_crit, control, cluster, 0], queue_size=1)
		#rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, callback4, [knn, track_window, roi_hist, term_crit, control, cluster, 0], queue_size=1)
		
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
	#print(len(point_cloud_all))


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
	#print(len(keypoint_cloud))


def callback4(data, args):
	global pose_camara
	global point_cloud_all
	global keypoint_cloud
	
	#|--- Extraer imagen comprimida ---|#
	np_arr = np.fromstring(data.data, np.uint8)
	#|--- Codificar en formato opencv() ---|#
	img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	#|--- imagen ---|#
	if img_np is not None:
		#|--- Main ---|#
		distancia, args[1], args[2], args[4], args[5]  = main(img_np, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
	
		#print("|--- Camara ---|")
		cX1, cY1 = pose_camara.position.x, pose_camara.position.y
		#print(pose_camara)
		#print("|--- Cloud ---|")
		#punto = point_cloud_all[0]
		#x2, y2 = punto[0], punto[1]
		x2, y2 = distancia[0], distancia[1]
		Dx2, Dy2 = (x2*cX1), (y2*cY1)
		#print(point_cloud_all[0])
		#print("|--- Keypoints ---|")
		#print(keypoint_cloud[0])
	
		#print("|--- Raiz cuadrada ---|")
		raiz = math.sqrt((Dx2-cX1)**2)+((Dy2-cY1)**2)#0.008058467348075563
		
		#|--- Publicar resultado de analisis ---|#
		talker(raiz)


def talker(distancia):
	#|--- Generar publisher ---|#
	pub = rospy.Publisher('Distancia', Float32, queue_size=1)
	#rate = rospy.Rate(1) # 10hz
	if not rospy.is_shutdown():
		#|--- Publicar resultado ---|#
		pub.publish(distancia)
		#print("|--- Distancia ---|")
		#print(distancia)
		#rate.sleep()


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
    file = r_archivo("/home/cesar/catkin_ws/src/tracking/src/Descriptores/Botella.txt")
    file2 = r_archivo("/home/cesar/catkin_ws/src/tracking/src/Descriptores/NoBotella.txt")
    #|--- variables locales ---|#	
    trainData = []
    responses = []
    #|---- Organizar informacion ----|#
    for i in file:
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

def convol(img, control):
    if control == 1:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    if control == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img2 = cv2.filter2D(img,-1,kernel)
    img = cv2.add(img,img2)
    return img

def mejorar_imagen(img):
    #img = convol(img, 1)
    img = ecualizar(img)
    return img

def ecualizar(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def orb_features(img):
    #|--- Orb features ---|#
    orb = cv2.ORB_create(nfeatures=650, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)    
    kp, des = orb.detectAndCompute(img, None)
    #img2 = cv2.drawKeypoints(img,kp, None,color=(0,255,0), flags=0)
    kp = np.array([kp[idx].pt for idx in range(0, len(kp))]).astype(int)
    #cv2.imshow("imagen2",img2)
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
    minx, maxx, miny, maxy = 1000, 0, 1000, 0
    #|--- Buscar puntos maximos y minimos ---|#
    for i in position:
        point = kp[i]
        roi_kp.append(point)
        roi_des.append(des[i])
        if minx > point[0]:
            minx = point[0]
        if maxx < point[0]:
            maxx = point[0]
        if miny > point[1]:
            miny = point[1]
        if maxy < point[1]:
            maxy = point[1]
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
    resultados, botella = [], []
    #|--- Organizar cluster ---|#
    for i in cluster:
        #|--- reduccion a 50 descriptores ---|#
        descriptor = reduccion(i[9])
        #|--- estandarizar al knn ---|#
        newcomer = np.array(descriptor).astype(np.float32)
        #|--- aplicar knn ---|#
        ret, results, neighbours, dist = knn.findNearest(newcomer, 21)
        #|--- Almacenar resultados ---|#
        resultados.append(resultado_knn(results))
    #|--- Analisar resultados ---|#
    for i,j in enumerate(resultados):
        if j == 0:
            botella.append(cluster[i])
    #|--- Retornar ---|#   
    return botella

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
    hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Coca", hsv_roi)
    #|--- Make a point window ---|#
    centro = cluster[6]
    w, h = cluster[4], cluster[5]
    x, y, w, h = centro[0], centro[1], w, h
    track_window = (x, y, w, h)
    #|--- Make a mask ---|#
    mask = mascara(hsv_roi.copy())
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,179])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    #|--- Retornar ---|#
    return track_window, roi_hist, cluster


def sort(cluster):
    #|--- Ordenamiento por tamano de cluster ---|#
    cluster.sort(key = lambda x: x[7], reverse = True)
    #|--- Retornar ---|#
    return cluster


def mascara(img):
    #|--- Binarizar imagen ---|#
    ret, mask = cv2.threshold(img[:,:,1],127,255,cv2.THRESH_BINARY)#_INV
    #cv2.imshow("ASD",mask)
    #|--- Retornar ---|#
    return mask


def seguimiento(frame, roi_hist, track_window, term_crit ):
    #|--- Convertir imagen a formato HSV ---|#
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #|--- Calculate backprojection ---|#
    dst = cv2.calcBackProject([frame],[0],roi_hist,[0,179],1)
    cv2.imshow("hist", dst)
    #|--- Calculate a new location ---|#
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #|--- Draw a new region ---|#
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    x, y, w, h = track_window
    x1, y1, x2, y2 = (x-int(w/2)), (y+int(h/2)), (x+int(w/2)), (y-int(h/2)) 
    frame2 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    #|--- Retornar ---|#
    return frame2, track_window

def perdida_objeto(track, track_ant):#|--- No reconoce el if correctamente ---|#
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
	track_ant = track_window
	#|--- Mejoramiento de la imagen ---|#
	tiempo = t.time()
	img = mejorar_imagen(imagen.copy())
	tiempofinal = t.time() - tiempo
	print tiempofinal		
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
			print("Lista vacia.")
            		return 0, track_window, roi_hist, True, []
		#|--- Pre-Track ---|#
        	track_window, roi_hist, cluster = preparar_track(result)
        	control = False
	#|--- Tracking ---|#
	track_img, track_window = seguimiento(img.copy(), roi_hist, track_window, term_crit)
	control = perdida_objeto(track_window, track_ant)
	cv2.imshow("Imagen final", track_img)
	cv2.waitKey(1)
	#|--- Retornar ---|#
	return cluster[6], track_window, roi_hist, control, cluster
#|--------------------------------------------------- Inicio ---------------------------------------------------|#
if __name__ == '__main__':
	#|--- Ejecutar hilo ---|#
	listener()
    	cv2.destroyAllWindows()
	print("|--- Fin ---|")
