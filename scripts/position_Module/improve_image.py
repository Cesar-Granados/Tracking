#!/usr/bin/env python
import cv2
import numpy as np
class improve_image:
	def __init__(self):
		self.empty_image = True
		self.BGR_image = None
		self.HSV_image = None
		self.Gray_image = None
		self.ORB_keypoints = None
		self.ORB_descriptors = None
	

	def init_BGR(self, image):
		if image is None:
			print("Image given is emtpy.")
			return None
		self.empty_image = False
		self.BGR_image = image
		return


	def init_HSV(self):
		if self.empty_image is True:
			print("self.BGR_image is empty.")
			return None
		#--- Try to convert to HSV format ---#
		HSV = cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2HSV)
		if HSV is None:
			print("Error to assign the HSV format.")
			return None
		self.HSV_image = HSV
		return


	def init_GRAY():
		if self.empty_image is True:
			print("self.BGR_image is empty.")
			return None
		#--- Try to convert to GRAY format ---#
		GRAY = cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)
		if GRAY is None:
			print("Error to assign the GRAY format.")
			return None
		self.Gray_image = GRAY
		return


	def orb_features(self):
		if self.empty_image is True:
			print("self.BGR_image is empty.")
			return None
		#--- Obteniendo ORB ---#
		orb = cv2.ORB_create(nfeatures=650, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)
		kp, des = orb.detectAndCompute(self.BGR_image, None)
		#--- Obtener unicamente la parte entera de los keypoints ---#
		kp = np.array([kp[idx].pt for idx in range(0, len(kp))]).astype(int)
		#--- Update state ---#
		self.ORB_keypoints = kp
		self.ORB_descriptors = des
		return


	def equalize_image(self):
		if self.HSV_image is not None:
			self.HSV_image[:,:,2] = cv2.equalizeHist(self.HSV_image[:,:,2])
		else:
			print("Error en ecualizacion: ", self.HSV_image)
		#--- Update the state ---#
		if self.HSV_image is not None:
			self.BGR_image = cv2.cvtColor(self.HSV_image, cv2.COLOR_HSV2BGR)
		else:
			print("Error en conversion HSV-BGR: ", self.HSV_image)
		if self.Gray_image is not None:
			self.Gray_image = cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)
		return

	
	def promedio(self, list,n):
    		k = 0
    		for i in list:
        		k = k + i
		k = k/len(list)
		return [k,n]


	def mediacolors(self, imagen):
    		rojo, naranja, amarillo, verde = [],[],[],[]
    		azul, morado, rosa = [],[],[]
    
    		rojo = np.vstack((imagen[0:10],imagen[169:179]))
    		naranja = imagen[11:25]
    		amarillo = imagen[26:35]
    		verde = imagen[36:70]
    		azul = imagen[71:129]
    		morado = imagen[130:139]
    		rosa = imagen[140:168]

    		colors = []
    		colors.append(self.promedio(rojo,0))
    		colors.append(self.promedio(naranja,1))
    		colors.append(self.promedio(amarillo,2))
    		colors.append(self.promedio(verde,3))
    		colors.append(self.promedio(azul,4))
    		colors.append(self.promedio(morado,5))
    		colors.append(self.promedio(rosa,6))

    		colors.sort(key= lambda x:x[0][0], reverse=True)
    		return colors[0]


	def mascara(self,imagen,colores):
    		#|--- Mascara de rojos hsv ---|#
    		if colores[1] == 0:
        		low, upp = (170,100,100),(180,255,255)
        		low2, upp2 = (0,100,100),(10,255,255)
       			imagen1 = cv2.inRange(imagen,low, upp)
        		imagen2 = cv2.inRange(imagen,low2, upp2)
			imagen3 = cv2.add(imagen1,imagen2)
        		return imagen3
    		#|--- Mascara naranja|#
    		if colores[1] == 1:
        		low, upp = (11,100,100),(25,255,255)
        		return cv2.inRange(imagen,low, upp)
    		#|--- Mascara amarillo ---|#
    		if colores[1] == 2:
        		low, upp = (26,100,100),(35,255,255)
        		return cv2.inRange(imagen,low, upp)
    		#|--- Mascara verde ---|#
    		if colores[1] == 3:
        		low, upp = (36,100,100),(70,255,255)
        		return cv2.inRange(imagen,low, upp)
		#|--- Mascara azul ---|#
    		if colores[1] == 4:
        		low, upp = (71,100,100),(129,255,255)
        		return cv2.inRange(imagen,low, upp)
    		#|--- Mascara morado ---|#
    		if colores[1] == 5:
        		low, upp = (130,100,100),(139,255,255)
        		return cv2.inRange(imagen,low, upp)
    		#|--- Mascara rosa ---|#
    		if colores[1] == 6:
        		low, upp = (140,100,100),(168,255,255)
        		return cv2.inRange(imagen,low, upp)


	def calculo(self, image):
    		histograma = cv2.calcHist([image],[0],None,[180],[0,179])
    		histograma = cv2.normalize(histograma,  None, 0, 180, cv2.NORM_MINMAX)
    		colores = self.mediacolors(histograma)
    		mask = self.mascara(image, colores)
    		return mask, colores


	def tlog(self, hu):
    		if hu[0] > 0:
    			hu[0] = -np.sign(hu[0]) * np.log10(np.abs(hu[0]))
    		if hu[1] > 0:
			hu[1] = -np.sign(hu[1]) * np.log10(np.abs(hu[1]))
    		if hu[2] > 0:
    			hu[2] = -np.sign(hu[2]) * np.log10(np.abs(hu[2]))
    		if hu[3] > 0:
    			hu[3] = -np.sign(hu[3]) * np.log10(np.abs(hu[3]))
    		if hu[4] > 0:
    			hu[4] = -np.sign(hu[4]) * np.log10(np.abs(hu[4]))
    		if hu[5] > 0:
    			hu[5] = -np.sign(hu[5]) * np.log10(np.abs(hu[5]))
    		if hu[6] > 0:
    			hu[6] = -np.sign(hu[6]) * np.log10(np.abs(hu[6]))
    		return hu


	def Hu(self, image):
		mask, color = self.calculo(image)
    		momentos = cv2.moments(mask)
    		humoments = cv2.HuMoments(momentos)
    		humoments = self.tlog(humoments)
    		humoments = np.vstack((humoments,color[0]))
    		humoments = humoments.flatten()
		return humoments
