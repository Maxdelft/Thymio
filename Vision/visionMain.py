import numpy as np
import cv2
import matplotlib.pyplot as plt
from camera_pose import pose

def imgPreprocessing(img):
    '''
    Performs suitable preprocessing of img
    '''
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Smoothening
    img = cv2.bilateralFilter(img,50,150,150)
    # Thresholding
    _,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    # Denoiseing
    kernel = np.ones((20,20),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img.astype(np.uint8)

def getLimitDistance(img):
    '''
    Gets largest possible distance between points in img
    '''
    p_min = np.array([0,0])
    p_max = np.array([img.shape[0],img.shape[1]])
    return np.mean(np.sqrt((p_min-p_max) ** 2))

def getPoints(contours,i = 0):
    '''
    Returns points corresponding to feature i
    '''
    pointx = []
    pointy = []
    for point in contours[0][i]:
        pointy.append(point[0][1])
        pointx.append(point[0][0])
    return np.array(pointx),np.array(pointy)

def contourSize(pointx,pointy):
    '''
    Caculated maximum distance between 2 points in a contour
    '''
    x_max = np.amax(pointx)
    x_min = np.amin(pointx)
    y_max = np.amax(pointy)
    y_min = np.amin(pointy)
    return np.array([x_min,y_min]), np.array([x_max,y_max])

def getLargestContour(contours,contourSizeLimit):
    '''
    Returns points of the largest found contour
    '''
    N_cont = len(contours[0])
    distances = np.empty(N_cont)
    for i in range(N_cont):
        pointx,pointy = getPoints(contours,i)
        p_min,p_max = contourSize(pointx,pointy)

        distances[i] = np.mean(np.sqrt((p_min-p_max) ** 2))

        # Ignore contours including image corners
        tol = 30
        if contourSizeLimit < distances[i] + tol:
            distances[i] = np.nan
    max_ind = np.where(distances == np.amax(distances))[0][0]

    return getPoints(contours,max_ind)

def findCorners(pointsx,pointsy):
    '''
    Finds extremevalues of points scattered on a parallelogram
    '''
    p_sum = pointsx+pointsy
    p_diff = pointsy-pointsx

    p_min_idx = np.argmin(p_sum)
    p_max_idx = np.argmax(p_sum)
    p_diffmin_idx = np.argmin(p_diff)
    p_diffmax_idx = np.argmax(p_diff)
    topR = np.array([pointsx[p_diffmin_idx],pointsy[p_diffmin_idx]])
    topL = np.array([pointsx[p_min_idx],pointsy[p_min_idx]])
    botR = np.array([pointsx[p_max_idx],pointsy[p_max_idx]])
    botL = np.array([pointsx[p_diffmax_idx],pointsy[p_diffmax_idx]])
    return np.array([botL,botR,topR,topL],dtype = np.float32)

def getMap(img):
    '''
    Returns corners of map from camera img
    '''
    # Calculate contours
    contours = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Returns contours coordinates with maximum distance
    contourSizeLimit = getLimitDistance(img)                # To ignore contour of image edges
    pointsx,pointsy = getLargestContour(contours,contourSizeLimit)
    return findCorners(pointsx,pointsy)

def getTransform(img_IN,img,src_points,y_size = 300):
    '''
    Crops and performs 4-point perspective transform on img based on points in src_points.
    Resulting img resolution can be specified using y_size
    '''
    scalexy =  1260/891             # Actual physical map scale in mm 

    x_size = int(y_size * scalexy)

    dst_points = np.array([[0,y_size],[x_size,y_size],[x_size,0],[0,0]], dtype = np.float32)

    T = cv2.getPerspectiveTransform(src_points, dst_points)
    warp_map = cv2.warpPerspective(img, T, (x_size, y_size))
    warp_img = cv2.warpPerspective(img_IN, T, (x_size, y_size))
    return warp_img,warp_map,T

def doPerspectiveTransform(img,T,y_size = 300):
    scalexy =  1260/891
    x_size = int(y_size * scalexy)
    return cv2.warpPerspective(img, T, (x_size, y_size))

def plotVision(img,img_in,src):
    fig,ax = plt.subplots(1,2,figsize = (15,15))
    ax[0].imshow(img)
    ax[1].imshow(img_in)
    ax[0].scatter(src[:,0],src[:,1],marker = 'x',linewidth = 50)
    plt.show()

def getTranformedMap(img_IN,y_scale):
    '''
    Process input image and returns global map with corresponding transform
    '''
    img = imgPreprocessing(img_IN)
    src_points = getMap(img)
    return  getTransform(img_IN,img,src_points,y_scale)

def findTemplate(img,template):
    '''
    Returns coordinates and size of goal
    Match template code from:
    https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
    '''
    w, h = template.shape[::-1]
    # Apply template Matching
    method = 'cv2.TM_CCOEFF'
    method_int = eval(method)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img,template,method_int)
    max_loc = cv2.minMaxLoc(res)[3]
    topL = max_loc
    botR = (topL[0] + w, topL[1] + h)
    midx = max_loc[0]+int(w/2)
    midy = max_loc[1]+int(h/2)
    mid = (midx,midy)
    return mid, topL,botR


def findTemplateSIFT(img,template):
    '''
    Returns coordinates and size of goal
    SIFT template code from:
    https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(template,None)
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    N = len(matches)
    x_pos = 0
    y_pos = 0
    for i in range(N):
        indx = matches[i].queryIdx
        testGoal = keypoints_1[indx].pt
        x_pos += testGoal[0] / N
        y_pos += testGoal[1] / N


    return (x_pos,y_pos)

def getGoal(warp_img,templateFileName):
    '''
    Returns position of global navigation goal
    '''
    template_img = cv2.imread(templateFileName,cv2.IMREAD_GRAYSCALE)
    return findTemplateSIFT(warp_img,template_img)

def filterGoalNoise(warp_map,goalPos):
    '''
    Removes potential disturbance of goal visual from binary map
    '''
    for i in range(goalPos[1][1]-10,goalPos[2][1]+10):
        for j in range(goalPos[1][0]-10,goalPos[2][0]+10):
            warp_map[i,j] = 255
    return warp_map

def filterStartNoise(warp_map,startPos, size = 25):
    '''
    Removes potential disturbance of thymio visual from binary map
    '''
    iterx = np.arange(startPos[1]-size,startPos[1]+size, dtype = int)
    itery = np.arange(startPos[0]-size,startPos[0]+size, dtype = int)

    for i in iterx:
        for j in itery:
            warp_map[i,j] = 255
    return warp_map

def getNavigationMap(img,templateFileName = 'Vision/Images/template_Placeholder.png',y_scale = 300):
    """
    Main method.
    returns transformed map, start, goal and rotational transform
    """
    warp_img,warp_map,T  = getTranformedMap(img,y_scale)
    goalPos = getGoal(warp_img,templateFileName)
    startPos = pose(warp_img,True)[0:2]

    warp_map = filterStartNoise(warp_map,startPos)
    return warp_map, goalPos, startPos,T