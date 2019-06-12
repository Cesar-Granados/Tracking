from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from matplotlib import pyplot as plt
import cv2

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

def seleccionar_object(objeto,img):
    if len(objeto) == 0:
        print("No existen objetos.")
        print("|---- saliendo seleccionar_object ----|")
        return img, None
    #|---- local var ----|#
    botella = cv2.imread("Imagenes-Objeto/Botella/img6.png")
    hist_botella, res_hist = [], []
    k = 0
    #|---- Diff Histogram ----|#
    for channel in range(3):
        hist2 = cv2.calcHist([botella], [channel], None, [256], [0, 255])
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        hist_botella.append(hist2)
    
    for j in objeto:
        #|---- var de ciclo ----|#
        hist_frame = []
        #|---- crop img ----|#
        img2 = roi(j,img.copy())
        #|----  ----|#
        for channel in range(3):
            hist = cv2.calcHist([img2],[channel], None, [256], [0, 255])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hist_frame.append(hist)
        
        for i in hist_frame:
            if k >= 3: k =0
            res_hist.append(cv2.compareHist(hist_botella[k], i, cv2.HISTCMP_CORREL))
            k = k+1
    #|---- hitograma mas parecido ----|#
    R,G,B = 0,0,0
    control,position = 0,0
    for k,i in enumerate(objeto,0):
        comp_hist1, comp_hist2 = 0,0
        #||#
        if R < res_hist[control]:
            comp_hist2 = comp_hist2+1
        else:
            comp_hist1 = comp_hist1+1
            
        if G < res_hist[(control+1)]:
            comp_hist2 = comp_hist2+1
        else:
            comp_hist1 = comp_hist1+1
            
        if B < res_hist[(control+2)]:
            comp_hist2 = comp_hist2+1
        else:
            comp_hist1 = comp_hist1+1
        #||#
        if comp_hist1 < comp_hist2:
            R,G,B = res_hist[control],res_hist[control+1],res_hist[control+2]
            position = k
            control = control+3
        else:
            control = control+3
    #||#
    cl = objeto[position]
    img2 = roi(cl,img.copy())
    print("|---- saliendo seleccionar_object ----|")
    return img2, cl

def entrenar_knn(trainData,responses):
    #|------------------------------------------------|#
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    print("|--- Saliendo entrenar_knn ---|")
    return knn

def clasificar(des, knn):
    newcomer = reduccion(des)
    newcomer = np.array(newcomer).astype(np.float32)
    cero,uno = 0,0
    #|------------------------------------------------|#
    ret, results, neighbours, dist = knn.findNearest(newcomer, 17)
    #|---- iteracion de respuestas ----|#
    for i in results:
        if i == 0:
            cero = cero +1
        if i == 1:
            uno = uno+1
    print("|--- Saliendo clasificar ---|")
    if uno > cero:
        return 1
    else:
        return 0

def reduccion(descriptors):
    size = len(descriptors)
    ratio = int(size/50)
    new_descriptors = []
    w = 1
    for j in descriptors:
        
        if w == (ratio):
            new_descriptors.append(j)
            w = 1
            continue
        w = w+1
    return new_descriptors

def f_match(img1, orb):
    kp1, des1 = orb.detectAndCompute(img1, None)
    pts = np.array([kp1[idx].pt for idx in range(0, len(kp1))]).astype(int)
    return pts, des1

def ecualizar(img):
    #|Separar canales de la imagen|#
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img

def convol(img, control):
    if control == 1:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    if control == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img2 = cv2.filter2D(img,-1,kernel)
    img = cv2.add(img,img2)
    return img

def pro_img(img):
    img = ecualizar(img)
    img = convol(img, 1)
    #|----------------------------------------------------|#
    #print("|--- pro_img ---|")
    return img

def mascara(hsv, control):
    if control == 1:
        rojo_b1, rojo_a1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        rojo_b2, rojo_a2 = np.array([160, 100, 100]), np.array([179, 255, 255])
        #|------|#
        mask1 = cv2.inRange(hsv, rojo_b1, rojo_a1)
        mask2 = cv2.inRange(hsv, rojo_b2, rojo_a2)
        #|------|#
        return cv2.add(mask1, mask2)
    if control == 2:
        rojo_b1, rojo_a1 = np.array([0, 0, 255]), np.array([130, 130, 255])
        #|------|#
        mask1 = cv2.inRange(hsv, rojo_b1, rojo_a1)
        #|------|#
        return mask1
    return 0
def re_size(img,proporcion):
    alto, ancho, _ = img.shape
    #|---------------------------------|#
    alto = int(alto*proporcion)
    ancho = int(ancho*proporcion)
    #|---------------------------------|#
    #print("|--- re_size ---|")
    return cv2.resize(img,(ancho,alto))

def n_blobs(points, desc, ms, img, knn):
    #|--- Meanshif var ---|#
    bandwidth = estimate_bandwidth(points, quantile=0.1, n_samples=500)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    ms.fit(points)
    labels = ms.labels_
    cluster_center = ms.cluster_centers_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    #|--- local var ---|#
    cl, objeto = [], []
    #|---------------------------------------|#
    for k, c in enumerate(cluster_center, 0):
        #punto = (int(c[0]),int(c[1]))
        cl = roi_section(k, points, desc, labels)
        if cl[4] > 50:
            result = clasificar(cl[5],knn)
            #|--- Guardar descriptores ---|#
            #w_archivo(cl, k)
        if result == 0:
            objeto.append(cl)
        else:
            print("Puntos insuficientes: ",cl[4]) 
            
    img2, clfin = seleccionar_object(objeto, img)
    #|--- Conver format color in HSV ---|#
    img2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) 
    return img2, clfin

def roi(objeto,img2):
    w = objeto[1] - objeto[0]
    h = objeto[3] - objeto[2]
    if w < 15: w = 15
    if h < 15: h = 15
    img2 = img2[objeto[2]:objeto[2]+h,objeto[0]:objeto[0]+w]
    return img2

def roi_section(n_cluster, points, desc, labels):
    min_x,  max_x = 1000, 0
    min_y, max_y = 1000, 0
    size = 0
    text_desc = []    
    #|--------------------------------------------|#
    for i in labels:
        if i == n_cluster:
            c = indices(labels.tolist(), i)
            pt = points[c[size]]
            des = desc[c[size]]
            text_desc.append(des)
            #|------|#
            if min_x > pt[0]:
                min_x = pt[0]

            if max_x < pt[0]:
                max_x = pt[0]

            if min_y > pt[1]:
                min_y = pt[1]

            if max_y < pt[1]:
                max_y = pt[1]
            #|------|#
            size = size+1
    #|---------------------------------------------|#
    cl = [min_x, max_x, min_y, max_y, size, text_desc]
    print("|--- Saliendo roi_section ---|")
    return cl

def indices( lista, value):
    return [i for i,x in enumerate(lista) if x==value]

def seguimiento(frame, roi_hist, track_window, term_crit ):
    #|--- Calculate backprojection ---|#
    dst = cv2.calcBackProject([frame],[0,1],roi_hist,[0, 179, 0, 255],1)
    #|--- Calculate a new location ---|#
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #|--- Draw a new region ---|#
    x, y, w, h = track_window
    frame2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    return frame2, track_window

def preparar_track(roi, cl):
    #|--- Make a point window ---|#
    x, y, w, h = cl[0], cl[2], (cl[1] - cl[0]), (cl[3] - cl[2])
    track_window = (x, y, w, h)
    #|--- Make a mask ---|#
    mask = mascara(roi, 1)
    roi_hist = cv2.calcHist([roi], [0, 1], mask, [180, 256], [0,179, 0,255])
    #plt.imshow(roi_hist)
    #plt.show()
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    #|--- Establecer condiciones de paro|#
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    return track_window, roi_hist, term_crit

#|---------------------------------------------------|#
cap = cv2.VideoCapture("Videos/Oorion1.avi")
orb = cv2.ORB_create()
ms = MeanShift()
#|---------------------------------------------------|#
trainData, responses = conversion_matriz()
knn = entrenar_knn(trainData, responses)
#|---------------------------------------------------|#
track_window, roi_hist, term_crit = (),0,0
control, k = True, 0
#|---------------------------------------------------|#
while cap.isOpened():
    ret, frame = cap.read()
    #|-----------------------------------------------|
    if not ret:
        print("Fin del stream.")
        break
    img1 = pro_img(frame.copy())
    pts, desc = f_match(img1, orb)
    if control == True:
        #|---- ROI and points ----|#
        img2, cl = n_blobs(pts, desc, ms, img1,knn)
        if cl != None:
            track_window, roi_hist, term_crit = preparar_track(img2,cl)
            control = False
    track_img, track_window = seguimiento(frame, roi_hist, track_window, term_crit)
    cv2.imshow("Frame",frame)
    cv2.imshow("Imagen final",track_img)
    cv2.waitKey(100)
print("|--- Fin ---|")