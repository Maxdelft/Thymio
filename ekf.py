#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:13:55 2021

@author: maximilianvanamerongen

the state is X_k is defined as: [x_k, y_k,theta_k,v_k, omega_k]
        with: x_k,y_k := x,y-position
        theta_k       := orientation
        v_k           := linear velocity
        omega_k       := angular veloctiy
        
the measurement Z_k is defined to be equal to the state X_k as: [x_k, y_k, theta_k,v_k, omega_k]
       where: 
           x_k,y_k, theta_k are measured by the camera system
           v_k, omega_k are calculated from the measured right and left wheel velocites 
                
"""

# https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb

#from filterpy.kalman import ExtendedKalmanFilterKalmanFilter
import numpy as np
import math 



def trans_vel(v_measured):
    '''
    Input : Measured wheel velocity in Thymio units   
    Output: Measured wheel velocity in m/s
    '''
    ratio      = 0.0002735 * 2
    v_ms = v_measured*ratio 
    return v_ms 

def initialize_kalman():
    '''
    Q := Measurment noise matrix
       given in: 
       diag([m, m, rad, m/s, rad/s])**2
    R := Measurement noise matrix
    given in: 
       diag([m, m, rad, m/s, rad/s])**2

    x0 := initiale state is set to zero. Can be improved by making use of first measurement of aruco
    P0 := initial covariance matrix is set to identity
    '''

    Q = np.diag([0.01764, 0.008316,0.043398 * 3, 0.003, 0.047])**2#np.diag([0.01, 0.01, 0.17454, 0.00244539, 0.00114760])**2 
    R = np.diag([0.2*0.01764, 0.2*0.008316,0.2*0.043398, 0.003, 0.047])**2#np.diag([0.002,0.002, 0.0436,0.00244539, 0.00114760])**2

    P0pred = np.eye(5) * 0.1
    P0up   = np.eye(5) * 0.1

    return  Q, R, P0pred, P0up
    




def cal_pix2mm(input_image):
    # Input: input_image := image of your map
    # Transform pixel to mm 
    
    global pix2m_y, pix2m_x
    
    height = input_image.shape[0] 
    width  = input_image.shape[1]
    
    pix2m_y = 0.891/height
    pix2m_x = 1.260/width
    
    return 0

 

def calculate_vel(V_r,V_l,L = 0.095):
    
    '''
    Task: This function calculatates the linear and angular velcoities based on the measured wheelspeeds
    
    input:
        V_r,V_l := measured wheel velocity of right and left wheel in mm/s 
        L       := wheel base of the robot  
    
    output:
        u containing: 
        u[0] := v       := linear velocity
        u[1] := omega   := angular velcoity     
        
     '''
    v     = (V_r + V_l)/2 
    omega = (V_r - V_l)/L
    
    u = np.array([v,omega])
    return u.T



def jacob_h():
    '''
    
    Task: The jacobian of the measurement function is calculated.
    
    The measurment function is defined as:
        
        z_k = h(X_k) + v_k  
        
    with:
        
        z_k    := the measurment
        h(X_k) := nonlinear function that relates states to measurements
        v_K    := measurment noise
        
    In our case this function is equal to the linear function z_k = h*X_k+v_k:
        
        with h equal to the indenty function

    input:  -
         
    output: jacobian of meaurment function
    
    '''
    H_jacob = np.eye(5)
    
    return H_jacob



def jacob_f(theta,v,omega,delta_t):
    
    '''
    Task: Here the jacobian of the state transition function is calucalted. 
    
    The state transition function is defined as: 
        X_k+1 = f(X_k,u_k) + w_k
        
    with:
        X_k+1/X_k := the state at time step k+1/k
        u_k       := the input to the system (linear and angular velocity)
        w_k       := process noise
        
    
    In our case the state transition function is defined as: 
        
        X_k+1 = [1 0 0 0 0 
                 0 1 0 0 0 
                 0 0 1 0 0
                 0 0 0 0 0 
                 0 0 0 0 0]*X_k +
                 
                 [delta_t*cos(theta_k) 0
                  delta_t*sin(theta_k) 0
                  0                    delta_t
                  1                    0
                  0                    1]*[v_k
                                           omega_k]
    '''

    theta = theta - math.pi/2
  
    F_jacob = np.array([
                       [1.0, 0.0, -delta_t*math.sin(theta)*v, delta_t*math.cos(theta), 0.0],
                       [0.0, 1.0,  delta_t*math.cos(theta)*v, delta_t*math.sin(theta), 0.0],
                       [0.0, 0.0, 1.0, 0.0, delta_t],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0] 
                       ])
                  
    return F_jacob




def EKF_predict(X_up,P_up,delta_t,u, Qk):
    
    '''
    This function performs the state and covariance prediction step of the extended kalman filter according to: 
        
        X_k+1 := f(X_k, u_k)
        
    which in our case is equal to:
        X_k+1 = f_state*X_k_est + f_inputs*u
        
    The covariance predicition step is perfomed according to:
        P_k+1 = F_k*P_k_est*F^T+Q_k
        
        
    Input:
        theta   := orientation 
        delta_t := sampling time
        Xk_est  := estiamted state
        Pk_est  := estiamted covariance matrix
        u       := input with u[0] := linear velocity
                   input with u[1] := angular velocity
        Q_k     := process noise covariance matrix
    
    Output:
        Xkp1    := predicted state
        Pkp1    := predicted covariance matrix
    
    '''
    # !
    X_up = X_up.T
    u = np.array([u])

    v     = u[0,0]
    omega = u[0,1]
    theta = X_up[2]

    theta = theta - math.pi/2
    
    f_states = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
    
    f_inputs = np.array([
        [delta_t*math.cos(theta), 0],
        [delta_t*math.sin(theta), 0], 
        [0, delta_t],
        [1, 0],
        [0, 1],
        ])
    
    # Jacobian of the state transition equation
    F = jacob_f(theta,v,omega,delta_t) 
    # State prediciton:
    Xpred = f_states @ X_up + f_inputs @ u.T
    
    # Covariance matrix prediciton:
    Ppred = F @ P_up @ np.transpose(F) + Qk
    
    return Xpred.T, Ppred

def EKF_correct(Xpred,Ppred,zk,Rk):
    '''
    
    Task: This function performs the state and covaraince matrix correction step of the extended kalman filter according to:
        S_k    := H*P_k_pred*H^T+R
        K_k    := P_k_pred*H^T*S_k^-1          (Kalman Gain)
        X_k_up := X_k_pred + K_k*(z_k-h(X_k))  (updated state)
        P_k_up := (I-K_k*H_k)*P_k_pred         (updated covariance matrix)
    
    
    Input:
        Pk_pred:= Predicted Covariance Matrix
        Xk_pred:= Predicted State
        zk     := Measurement
        Rk     := Masurment Noise Covariance Matrix
        
    Output: 
        Pk_up  := Updated covaraiance matrix
        Xk_up  := Updated state matrix
        
    '''
    
    # Calculate jacobian
    Hk = jacob_h()
    
    # Innovation/ residual covariance
    S = (Hk @ Ppred @ np.transpose(Hk) + Rk).astype(float)
    
    # Kalman Gain:
    K = Ppred @ np.transpose(Hk) @ np.linalg.inv(S)
    
    # Innovations:
    e_y  = zk - Xpred 
    # Update state estimate: 
    Xup = (Xpred.T + K @ e_y.T).T
    
    # Update Covariance matrix: 
    Pup = (np.eye(5)-K @ Hk)@ Ppred
    
    return Pup, Xup



def EKF(thymio): 

    # Get inputs
    Ppred = thymio.Ppred
    Xpred = thymio.Xpred
    zk = np.append(thymio.pos_meas,thymio.vel_meas,axis = 1)

    # Measurement step)
    Pup, Xup = EKF_correct(Xpred,Ppred, zk,thymio.R)

    # Prediction step:
    Xpred, Ppred = EKF_predict(Xup,Pup,thymio.delta_t, thymio.u, thymio.Q)

    # Return values to thymio object
    thymio.kalman_prediction(Xpred,Ppred)
    pos_est = np.array([Xup[0,0:3]])
    vel_est = np.array([Xup[0,3:5]])
    thymio.kalman_update([pos_est,vel_est,Pup])


def kalman_measure(node,thymio):
    '''
    Calculates extended kalman filter pose estimations.
    Based on pose estimation from camera and measurments of wheel speed
    input:
        node object
        thymio object
    '''
    v_l, v_r = node.v.motor.left.speed , node.v.motor.right.speed     # Read motor speed
    V_r, V_l = trans_vel(v_l), trans_vel(v_r)                         # velocity is translated to m/s
    thymio.velocity_meas(calculate_vel(V_r,V_l))                      # velocity is written to thymio

    u_l, u_r = node.v.motor.left.target, node.v.motor.right.target
    U_l, U_r = trans_vel(u_l), trans_vel(u_r)

    thymio.input_update(calculate_vel(U_r,U_l))

    EKF(thymio)

        