import numpy as np
import math

class myThymio():
    '''
    Object containing functions and continuously updated variables related to the Thymio
    '''
    def __init__(self,startPos,startVel,kalman,delta_t):
    
        # Initialize Kalman filter variables
        self.pos_est = startPos 
        self.vel_est = startVel 

        self.pos_meas = startPos
        self.vel_meas = startVel

        self.Xpred = np.append(startPos,startVel,axis = 1)

        self.Q = kalman[0] 
        self.R = kalman[1]
        self.Ppred = kalman[2]
        self.Pup   = kalman[3]
        self.Xup  = np.empty((1,5))
        self.u  = np.empty((2,1))

        # Set sample time
        self.delta_t = delta_t

        # Initalize path
        self.current_point = 0
        self.rx = []
        self.ry = []

        # Local obstacle modes
        self.obstacleFound = False
        self.avoidingObstacle = False
        self.obstacle_iter = 0

        # Initilize vectors for position plot
        self.xPlot = np.empty(1,)
        self.yPlot = np.empty(1,)

    def input_update(self,input):
        """
        Updates input signal
        """
        self.u = input

    def camera_meas(self,meas):
        """
        Returns camera pose estimate if the aruco code is succesfully identified.
        else returns last pose estimate
        """
        if meas[0] != -1:           # If not NUll update, update measurements
            self.pos_meas = np.array([meas])
        else:
            self.pos_meas = np.array([self.Xpred[0][0:3]])

    def velocity_meas(self,meas):
        """
        Updates latest velocity measurement
        """
        self.vel_meas = np.array([meas])

    def kalman_update(self,meas):
        """
        Updates Kalman filter pose and covariance estimate
        """
        self.pos_est    = meas[0]  # update pose estiamte
        self.vel_est = meas[1]      # update velocity estiamte
        self.Pup     = meas[2]      # update covariance estiamte
    
    def kalman_prediction(self,Xpred,Ppred):
        """
        Updates Kalman filter prediction pose and covariance
        """
        self.Xpred = Xpred
        self.Ppred = Ppred

    def savePath(self,rx,ry):
        """
        Stores the calculated reference path
        """
        self.rx = np.array(rx)
        self.ry = np.array(ry)

    def updatePoint(self):
        """
        Increments 
        """
        self.current_point += 1

    def distance2path(self):
        """
        Calculates distance between thymio and next point in reference path
        """
        distx2 = (self.rx[self.current_point]-self.pos_est[0][0]) ** 2 
        disty2 = (self.ry[self.current_point]-self.pos_est[0][1]) ** 2
        self.distPoint = np.sqrt(distx2+disty2)

    def minDist2path(self):
        """
        Calculates the distance between Thymio and all points in reference path
        returns closest point
        """
        distx2 = (self.rx-self.pos_est[0][0]) ** 2 
        disty2 = (self.ry-self.pos_est[0][1]) ** 2
        dist = np.sqrt(distx2+disty2)
        self.closestPoint = np.argmin(dist)
        self.smallestDist = np.min(dist)


    def angle2point(self):
        """
        Calculates difference between current thymio orientation and the angle to next reference point in path
        """
        distx = (self.rx[self.current_point+1]-self.pos_est[0][0])   # !! +1 should give error at end!!!! 
        disty = (self.ry[self.current_point+1]-self.pos_est[0][1])
        angle = math.atan2(distx,disty)
        self.angle = np.mod(angle, 2*np.pi)
        self.measAngle = np.mod(self.pos_est[0][2], 2*np.pi)
        self.anglePoint = self.measAngle - self.angle

    def setSpeed(self,motorLeft,motorRight):
        """
        Sets thymio motors speed target
        """
        self.motorLeft  = motorLeft
        self.motorRight = motorRight

    def storePath(self,x,y):
        """
        Stores current and previous positions of the thymio
        """
        self.xPlot = np.append(self.xPlot,x)
        self.yPlot = np.append(self.yPlot,y)
        


    
  
