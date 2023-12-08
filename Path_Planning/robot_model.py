import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.qhull import tsearch

def robot_model(x,u,T):
    v = u[0]
    w = u[1]
    s = T
    c = 0
    for k in range(1,4):
        s = s + (-1)**(k) * w**(2*k)*T**(2*k+1)/math.factorial(2*k+1)
        c = c - (-1)**(k) * w**(2*k-1)*T**(2*k)/math.factorial(2*k)
    theta = x[2,0]
    S = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    if (abs(w)<1e-6):
        xp = x + np.matmul(S, np.array([[v*s], [v*c], [w*T]]))
    else:
        xp = x + np.matmul(S,(np.array([[v/w*(math.sin(w*T))], [v/w*(1-math.cos(w*T))], [w*T]])))
    return xp

def model_prediciton(u, x0, T):
    #x = []
    x_k = x0
    #x.append(x0.T)
    x = x0.T
    for k in range(np.shape(u)[0]):
        x_new = robot_model(x_k, u[k,:], T)
        x_new[2,0] = angle2stdrange(x_new[2,0])
        
        x_k = x_new
        #x.append(x_k.T)
        x = np.append(x,x_k.T,axis = 0)
    return x

def angle2stdrange(w):
#brings angle omega to intervall ]-pi, pi]
    while (w>math.pi):
        w = w-2*math.pi
    while (w<=-math.pi):
        w = w+2*math.pi
    return w

def inverse_kinematics(x,x_old,T):

    w = (1/T)*(angle2stdrange(x[2]-x_old[2]))
    S_theta = np.array([[math.cos(x_old[2]), -math.sin(x_old[2]), 0], [math.sin(x_old[2]), math.cos(x_old[2]), 0], [0, 0, 1]])
    s = T
    c = 0
    for k in range(1,4):
        s = s + (-1)**(k) * w**(2*k)*T**(2*k+1)/math.factorial(2*k+1)
        c = c - (-1)**(k) * w**(2*k-1)*T**(2*k)/math.factorial(2*k)

    x_diff = x-x_old
    x_diff[2] = angle2stdrange(x_diff[2])
    if (abs(w) > 1e-3):
        A = np.array([[math.sin(w*T)/w, 0 ,0], [0, (1-math.cos(w*T))/w, 0], [0, 0, T]])
        mem = np.matmul(np.linalg.inv(A),S_theta.T)
        u = np.matmul(mem,x_diff)
        #print(u)
    elif w != 0:
        A = np.array([[s, 0, 0], [0, c, 0], [0, 0, T]])
        mem = np.matmul(np.linalg.inv(A),S_theta.T)
        u = np.matmul(mem,x_diff)
        #print(u)
    else:
        u = np.matmul(S_theta.T,x_diff)
        u = (1/T)*u
    
    return u[0], u[2]
   
def motor_speed(v,w,R,L,factor):
    #wheelospeed in rad/s
    #factor : \omega = (\omega/R)*motor_speed
    Transform = np.array([[1/R, L/(2*R)],[1/R, -L/(2*R)]])
    u = np.array([[v],[w]])
    s_wheel = np.matmul(Transform, u)
    motor_right_target = s_wheel[0]*(R/factor)
    motor_left_target = s_wheel[1]*(R/factor)
    return motor_right_target[0], motor_left_target[0]


# mat = scipy.io.loadmat('input_lemniscate_100.mat')
# input = mat.get('u3')
# x0 = np.array([[0], [0], [0]])
# dt = 0.2
# x = model_prediciton(input, x0, dt)
# plt.plot(x[:,0], x[:,1], ".k")
# plt.ylim((-5,5))
# plt.xlim((-2,2))
# print('ahla')


