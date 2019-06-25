from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2
import math

#|---------------------------------------------------|#
def conversion_matriz():
    #|---- local var ----|#
    file = r_archivo("Descriptores/Botella.txt")
    file2 = r_archivo("Descriptores/NoBotella.txt")
    trainData = []
    responses = []
    #|---- construct data ----|#
    for i in file:
        for j in i:
            trainData.append(j)
            responses.append(0)
            
    for i in file2:
        for j in i:
            trainData.append(j)
            responses.append(1)
    #|---- format ----|#
    trainData = np.array(trainData).astype(np.float32)
    responses = np.array(responses).astype(np.float32)
    print("|--- Saliendo conversion_matriz ---|")
    return trainData, responses

def w_archivo(cl, k):
    reducido = reducir(cl[5],50)
    file = open("Prueba.txt", "a")
    file.write('\nCl'+str(k)+'_Pt'+str(cl[4])+'_'+'.jpg')
    for i in reducido:
        file.write('\n'+str(i))
    file.write('\nfin-cluster')
    #cv2.imwrite('Cl'+str(k)+'_Pt'+str(cl[4])+"_"+str(j)+'.jpg', img2)
    file.close()
    print("|--- Saliendo w_archivo ---|")
    return
def r_archivo(path):
    arrayMaster = []
    #|--- parseo ---|#
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
    print("|--- Saliendo r_archivo ---|")
    return arrayMaster

def comparar_resultados(cl, cl_fin):
    roi_img, roi_final = cl[10], cl_fin[10]
    if roi_final is None:
        cl_fin = cl
        return cl_fin
    #|--- verificar result ---|#
    if roi_img == None:
        return cl_fin
    #|--- variables locales ---|#
    mayor = 0
    #|--- comparar  ---|#
    for i in range(3):
        if roi_final[i] < roi_img[i]:
            mayor = mayor+1
    if mayor >= 2:
        cl_fin = cl
    return cl_fin

def seleccionar_imagen(cl, img):
    centro, img2 = cl[6], img.copy()
    #|--- Calcular marco del cluster ---|#
    top, left = centro[1]+int((cl[5]/2)), centro[0]-int((cl[4]/2))
    bottom, right = centro[1]-int((cl[5]/2)), centro[0]+int((cl[4]/2))
    #|--- Dibujar cluster ---|#
    img = cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),2)
    img = cv2.circle(img,(centro[0],centro[1]), 5, (0,255,255), -1)
    #|--- ROI ---|#
    img2 = img2[bottom:top, left:right]
    #|--- Variables de analisis ---|#
    imagen = cv2.imread("Imagenes-Objeto/Botella/img0.png")
    hist_botella, hist_frame, result = [], [], []
    #|--- Mejorar imagen ---|#
    imagen = re_size(imagen,0.1)
    imagen = ecualizar(imagen)
    #|--- Calcular histograma ---|#
    for channel in range(3):
        #|--- Botella estandarizada ---|#
        cal_histB = cv2.calcHist(imagen[:,:,channel], [channel], None, [256], [0, 255])
        cv2.normalize(cal_histB, cal_histB, 0, 255, cv2.NORM_MINMAX)
        hist_botella.append(cal_histB)
        #|--- Imagen extraida del frame ---|#
        cal_hisF = cv2.calcHist(img2[:,:,channel], [channel], None, [256], [0, 255])
        cv2.normalize(cal_hisF, cal_hisF, 0, 255, cv2.NORM_MINMAX)
        hist_frame.append(cal_hisF)
    
    for k,i in enumerate(hist_botella,0):
        result.append(cv2.compareHist(i,hist_frame[k], cv2.HISTCMP_CORREL))
    
    #|--- Agregar imagen comparada ---|#
    result.append(img2)
    cl[10] = result
    cv2.imshow("Botella", imagen)
    return img, cl

def entrenar_knn(trainData,responses):
    #|------------------------------------------------|#
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    print("|--- Saliendo entrenar_knn ---|")
    return knn

def reduccion(descriptores):
    size = len(descriptores)
    ratio = int(size/50)
    new_descriptores = []
    w = 0
    for i in range(50):
        new_descriptores.append(descriptores[w])
        w = w + ratio
        
    return new_descriptores

def orb_features(img):
    #|--- Orb features ---|#
    orb = cv2.ORB_create(nfeatures=650, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)    
    kp, des = orb.detectAndCompute(img, None)
    img2 = cv2.drawKeypoints(img,kp, None,color=(0,255,0), flags=0)
    kp = np.array([kp[idx].pt for idx in range(0, len(kp))]).astype(int)
    #cv2.imshow("imagen2",img2)
    return kp, des

def ecualizar(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def convol(img, control2):
    if control2 == 1:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    if control2 == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img2 = cv2.filter2D(img,-1,kernel)
    img = cv2.add(img,img2)
    return img

def mejorar_imagen(img):
    img = ecualizar(img)
    #img = convol(img, 1)
    return img

def re_size(img,proporcion):
    alto, ancho, _ = img.shape
    #|---------------------------------|#
    alto = int(alto*proporcion)
    ancho = int(ancho*proporcion)
    #|---------------------------------|#
    return cv2.resize(img,(ancho,alto))

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

def k_nearest(knn, cluster):
    #|--- Ajustar escriptores ---|#
    descriptores = cluster[9]
    descriptores = reduccion(descriptores)
    newcomer = np.array(descriptores).astype(np.float32)
    #|--- Ejecutar knn ---|#
    ret, results, neighbours, dist = knn.findNearest(newcomer, 31)
    #|--- Analizar resultados ---|#
    return resultado_knn(results)

def meanshift_cl(kp, des, img):
    #|--- variables locales ---|#
    cluster_final = [0,0,0,0,0,0,0,0,0,0,None]
    #|--- Meanshif var ---|#
    bandwidth = estimate_bandwidth(kp, quantile=0.1, n_samples=650)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True, min_bin_freq=2, cluster_all=False)
    ms.fit(kp)
    labels = ms.labels_
    lista_label = labels.tolist()
    cluster_center = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Clousters: ", n_clusters_)
    img3 = img.copy()
    #|--- Analizar cada cluster del frame ---|#
    for k,i in enumerate(cluster_center,0):
        centro = [int(i[0]), int(i[1])]
        cluster = max_min(lista_label,k, kp, des, centro)#|Problema al igualar en nullo.|#
        if cluster[4] <= 10 or cluster[5] <= 10:
            print("Cluster rechazado: ","Ancho:",cluster[4], " Alto:", cluster[5],)
            continue 
        img3, cluster = seleccionar_imagen(cluster, img3)
        cluster_final = comparar_resultados(cluster, cluster_final)
        
    return cluster_final

def max_min(label, cluster, kp, des, centro):
    position = indice(label,cluster)
    size = len(position)
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
    
    w, h = int(maxx - minx), int(maxy - miny)
    cluster = [minx, maxx, miny, maxy, w, h, centro, size, roi_kp, roi_des, []]
    return cluster

def indice( lista, value):
    return [i for i,x in enumerate(lista) if x==value]

def seguimiento(frame, roi_hist, track_window, term_crit ):
    #|--- Calculate backprojection ---|#
    dst = cv2.calcBackProject([frame],[0, 1, 2],roi_hist,[0,179, 0,255, 0,255],1)
    #|--- Calculate a new location ---|#
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #|--- Draw a new region ---|#
    x, y, w, h = track_window
    x1, y1, x2, y2 = (x-int(w/2)), (y+int(h/2)), (x+int(w/2)), (y-int(h/2)) 
    frame2 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,), 2)
    return frame2, track_window

def preparar_track(cluster):
    roi_img = cluster[10]
    hsv_roi = roi_img[3]
    cv2.imshow("Track", hsv_roi)
    hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_BGR2HSV)
    #|--- Make a point window ---|#
    centro = cluster[6]
    w, h = cluster[4], cluster[5]
    x, y, w, h = centro[0], centro[1], w, h
    track_window = (x, y, w, h)
    #|--- Make a mask ---|#
    roi_hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [180, 256, 256], [0,179, 0,255, 0,255])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    #|--- Establecer condiciones de paro|#
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    return track_window, roi_hist, term_crit

def activate(anterior, track):
    (x, y, _, _), (x1, y1, _, _) = track, anterior
    
    dst = int(math.sqrt((x1-x)**2+(y1-y)**2))
    
    if dst > 30:
        print("Solicitar knn: \n","Distancia: ",dst)
        return True
    return False
#|---------------------------------------------------|#
cap = cv2.VideoCapture("Videos/Oorion1.avi")
orb = cv2.ORB_create(nfeatures = 650)
ms = 0#MeanShift()
#|---------------------------------------------------|#
trainData, responses = conversion_matriz()
knn = entrenar_knn(trainData, responses)
#|---------------------------------------------------|#
track_window, roi_hist, term_crit = (),0,0
ant_track = 0
control = True
#|---------------------------------------------------|#
while cap.isOpened():
    ret, frame = cap.read()
    #|-----------------------------------------------|
    if not ret:
        print("Fin del stream.")
        break
    img = mejorar_imagen(frame.copy())
    if control is True:
        #|--- ORB --|#
        kp, des = orb_features(img)
        #|--- Meanshift ---|#
        cluster = meanshift_cl(kp, des, img)
        #|--- KNN ---|#
        result = k_nearest(knn, cluster)
        if result == 1:
            continue
        #|--- Pre-Track ---|#
        track_window, roi_hist, term_crit = preparar_track(cluster)
        control = False
    
    ant_track = track_window
    track_img, track_window = seguimiento(img.copy(), roi_hist, track_window, term_crit)
    #control = activate(ant_track, track_window)
    
    #cv2.imshow("Frame",frame)
    cv2.imshow("Imagen final", track_img)
    cv2.waitKey(100)
print("|--- Fin ---|")
'''
-Deja de existir imagen en el entorno.
-Comparacion de hitogramas.
-Clasificado de knn.
'''