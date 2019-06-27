from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2
import math
import time

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
    #img2 = cv2.drawKeypoints(img,kp, None,color=(0,255,0), flags=0)
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
    return img2

def mejorar_imagen(img):
    #img = convol(img, 1)
    img = ecualizar(img)
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
            
    return botella

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
    print("Clusters: ", n_clusters_)
    img3 = img.copy()
    #|--- Analizar cada cluster del frame ---|#
    tcl = time.time()
    for k,i in enumerate(cluster_center,0):
        centro = [int(i[0]), int(i[1])]
        cluster = max_min(lista_label,k, kp, des, centro, img.copy())#|Problema al igualar en nullo.|#
        if cluster[7] < 50 or cluster[4] <= 10 or cluster[5] <= 10:
            print("Cluster rechazado: ", "ancho:",cluster[4], " alto:", cluster[5], " points:",cluster[7])
            continue
        cluster_final.append(cluster)
    tcl_f = time.time() - tcl
    print("Timepo correlacion.", tcl_f)
    return cluster_final

def max_min(label, cluster, kp, des, centro, img):
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
    #|--- Calculo del ancho y alto del cluster ---|#
    w, h = int(maxx - minx), int(maxy - miny)
    #|--- ROI image ---|#
    top, left = centro[1]+int((h/2)), centro[0]-int((w/2))
    bottom, right = centro[1]-int((h/2)), centro[0]+int((w/2))
    imagen = img[bottom:top, left:right]
    #|--- Organizar caracteristicas ---|#
    cluster = [minx, maxx, miny, maxy, w, h, centro, size, roi_kp, roi_des, imagen]
    return cluster

def indice( lista, value):
    return [i for i,x in enumerate(lista) if x==value]

def seguimiento(frame, roi_hist, track_window, term_crit ):
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
    return frame2, track_window

def mascara(img):
    ret, mask = cv2.threshold(img[:,:,1],127,255,cv2.THRESH_BINARY)#_INV
    cv2.imshow("ASD",mask)
    return mask

def sort(cluster):
    cluster.sort(key = lambda x: x[7], reverse = True)
    return cluster

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
        tcl = time.time()
        cluster = meanshift_cl(kp, des, img)
        tcl_f = time.time() - tcl
        print("Tiempo cluster: ", tcl_f)
        #|--- KNN ---|#
        tknn = time.time()
        result = k_nearest(knn, cluster)
        tknn_f = (time.time()-tknn)
        print("Tiempo KNN: ", tknn_f)
        if result == None or len(result) == 0:
            continue
        #|--- Pre-Track ---|#
        track_window, roi_hist, term_crit = preparar_track(result)
        #control = False
    
    #ant_track = track_window
    track_img, track_window = seguimiento(img.copy(), roi_hist, track_window, term_crit)
    #control = activate(ant_track, track_window)
     
    cv2.imshow("Imagen final", track_img)
    cv2.waitKey(100)
print("|--- Fin ---|")
'''
-Deja de existir imagen en el entorno.
-Distancia
[minx, maxx, miny, maxy, w, h, centro, size, roi_kp, roi_des, imagen]
'''