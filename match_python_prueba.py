from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2

#|---------------------------------------------------|#
def seguimiento(frame,img,cl):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    w,h = cl[1]-cl[0], cl[3] - cl[2]
    x,y = int((cl[0]+w)/2), int((cl[2]+h)/2)
    track_window = track_window = ( x, y, w, h)
    #|---------------|#
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    #|-------------------|#
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    # Draw it on image
    print(track_window)
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    return img2

def conversion_matriz():
    #|---- local var ----|#
    file = r_archivo("Botella.txt")
    file2 = r_archivo("NoBotella.txt")
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
            if("fin-cluster" in line):
                arrayCLuster.append(numberArray)
                arrayMaster.append(arrayCLuster)
    print("|--- Saliendo r_archivo ---|")
    return arrayMaster

def seleccionar_object(objeto,img):
    if len(objeto) == 0:
        print("No existen objetos.")
        print("|---- saliendo seleccionar_object ----|")
        return img, objeto
    if len(objeto) == 1:
        print("Una imagen.")
        print("|---- saliendo seleccionar_object ----|")
        img2 = roi(objeto[0],img.copy())
        return img2, objeto[0]
    #|---- local var ----|#
    botella = cv2.imread("Botella\img6.png")
    hist_botella = []
    res_hist = []
    #|---- Diff Histogram ----|#
    for channel in range(3):
        hist2 = cv2.calcHist([botella], [channel], None, [256], [0, 256])
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_botella.append(hist2)
    
    for j in objeto:
        #|---- var de ciclo ----|#
        hist_frame = []
        #|---- crop img ----|#
        img2 = roi(j,img.copy())
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        #|----  ----|#
        for channel in range(3):
            hist = cv2.calcHist([img2],[channel], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_frame.append(hist)
            
        for k,i in enumerate(hist_frame,0):
            res_hist.append(cv2.compareHist(hist_botella[k], i, cv2.HISTCMP_CORREL))
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
            R,G,B = res_hist[control+1],res_hist[control+2],res_hist[control+3]
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
    newcomer = reducir(des,50) 
    #newcomer.append([107, 249, 16, 90, 121, 157, 82, 252, 139, 121, 52, 12, 174, 217, 78, 180, 133, 91, 215, 92, 199, 106, 250, 245, 191, 221, 21, 123, 90, 14, 139, 159])
    newcomer = np.array(newcomer).astype(np.float32)
    cero,uno = 0,0
    #|------------------------------------------------|#
    #print(len(newcomer),newcomer.dtype)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 30)
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

def cargar_carac(path,orb):
    i = 0
    points = []
    descript = []
    imgs = []
    while(True):
        path1 = path+str(i)+".jpg"
        i= i+1
        img = cv2.imread(path1)
        if img is None:
            print("carga finalizada.")
            break
        img = re_size(img,0.1)
        img = pro_img(img)
        kp, des = orb.detectAndCompute(img, None)
        imgs.append(img)
        points.append(kp)
        descript.append(des)
    print("|--- Saliendo carga_caract ---|")
    return imgs, points, descript

def reducir(des,min):
    ratio = int(len(des)/min)
    control = 0
    new_des = []
    for i in range(min):
        new_des.append(des[control])
        control = control + ratio
    print("|--- Saliendo reducir ---|")
    return new_des

def f_match(img1, imgs, kp2, des2, orb, img2=None):
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    
    FLANN_INDEX_LSH = 0
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches_desc = []
    for d in des2:
        matches = flann.knnMatch(np.asarray(d,np.float32),np.asarray(des1,np.float32),k=2)
        matches_desc.append(matches)
    
    x=list(map(lambda a: len(a),matches_desc))
   
    maximo = x.index(max(x))
    matches = matches_desc[maximo]
    kp2 = kp2[maximo]
    img2 = imgs[maximo]
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[i]=[1,0,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
                
    '''for i, k in enumerate(matches,0):
        for j, p in enumerate(k[i],0):
            print(kp1[p[i][j].queryIdx].pt,kp2[p[i][j].trainIdx].pt)'''
    #print(kp1[matches[0][0].trainIdx].pt)
    
    img3 = cv2.drawMatchesKnn(img2,kp2,img1,kp1,matches,None,**draw_params)
    pts = np.array([kp1[idx].pt for idx in range(0, len(kp1))]).astype(int)
    print("|--- Saliendo f_match ---|")
    return img3, pts, des1

def pro_img(img):
    #|----------------------------------------------------|#
    kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #|----------------------------------------------------|#
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img2 = cv2.filter2D(img, 0, kernel2)
    img = cv2.addWeighted(img, 0.7, img2, 0.3, 0)
    #|----------------------------------------------------|#
    print("|--- pro_img ---|")
    return img

def re_size(img,proporcion):
    alto, ancho, _ = img.shape
    #|---------------------------------|#
    alto = int(alto*proporcion)
    ancho = int(ancho*proporcion)
    #|---------------------------------|#
    print("|--- re_size ---|")
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
    cl = []
    objeto = []
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
            
    img2, wd = seleccionar_object(objeto, img)
    #|----------------------------------------|#
    #cv2.imshow("Coca-Cola2",img2)
    #cv2.waitKey(1000)
    print("|---- frame ----|")   
    return img2, wd

def roi(objeto,img2):
    w = objeto[1] - objeto[0]
    h = objeto[3] - objeto[2]
    if w < 15: w = 15
    if h < 15: h = 15
    img2 = img2[objeto[2]:objeto[2]+h,objeto[0]:objeto[0]+w]
    return img2

def roi_section(n_cluster, points, desc, labels):
    min_x = 1000
    max_x = 0
    min_y = 1000
    max_y = 0
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
#|---------------------------------------------------|#
#img2 = cv2.imread('img0.jpg') # trainImage
cap = cv2.VideoCapture("Videos/Oorion1.avi")
orb = cv2.ORB_create()
ms = MeanShift()
#|---------------------------------------------------|#
trainData, responses = conversion_matriz()
knn = entrenar_knn(trainData, responses)
images_, kpoints_, descriptors_ = cargar_carac("Caracteristicas/img",orb)
frame_ant = []
#|---------------------------------------------------|#
while cap.isOpened():
    ret, frame = cap.read()
    #|-----------------------------------------------|
    if not ret:
        print("Fin del stream.")
        break
    img1 = pro_img(frame)
    img3, pts, desc = f_match(img1, images_,kpoints_, descriptors_, orb)
    img4, cl = n_blobs(pts, desc, ms, img1,knn)
    
    if len(cl) == 0:
        img4 = frame_ant[0]
        cl = frame_ant[1]
    else:
        frame_ant = [img4,cl]
        
    img5 = seguimiento(frame,img4,cl)
    
    cv2.imshow("img final",img5)
    cv2.waitKey(100)
print("|--- Fin ---|")
#|KD-Tree|#
#|Diferencia de histogramas|#
#|ROI - Clousters - Histogramas|#
#|Matriz de puntos - Match|#
