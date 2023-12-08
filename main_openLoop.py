import numpy as np
import cv2
from matplotlib import pyplot as plt
from myThymio import myThymio
from myThymio import myThymio
from tdmclient import ClientAsync
import sys
import os
import global_path as gp
import Path_planning.robot_model as rm
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'Vision'))
sys.path.insert(0, os.path.join(os.getcwd(), 'GlobalNavTest'))
sys.path.insert(0, os.path.join(os.getcwd(), 'Path_planning'))

#sys.path.insert(0, os.path.join(os.getcwd(), 'Localization/Global_Localization'))

from Vision.visionMain import getNavigationMap
from Vision.visionMain import doPerspectiveTransform
from Vision.camera_pose import pose
from ekf import *


def localObstacleAvoidance(node):
    #ommited for time reason
    pass

def kalman_measure(node,thymio): 
    v_l, v_r = node.v.motor.left.speed , node.v.motor.right.speed     # Read motor speed
    V_r, V_l = trans_vel(v_l), trans_vel(v_r)                         # velocity is translated to m/s
    thymio.velocity_meas(calculate_vel(V_r,V_l))                      # velocity is written to thymio

    u_l, u_r = node.v.motor.left.target, node.v.motor.right.target
    U_l, U_r = trans_vel(u_l), trans_vel(u_r)

    thymio.input_update(calculate_vel(U_r,U_l))

    EKF(thymio)


def makeVideo(thymio,frame_cropped):
    x_scale,y_scale =  gp.get_scale_factor_to_reality(frame_cropped)
    for i in range(len(thymio.rx)):
        plotPos = (int(thymio.rx[i] /x_scale * 1000),int(thymio.ry[i] / y_scale * 1000))
        edt_frame = cv2.circle(frame_cropped,
                    plotPos,
                    radius=1,
                    color = (0, 0, 255),
                    thickness = 2)
        if i == thymio.current_point:
            edt_frame = cv2.rectangle(frame_cropped,
                    (int(plotPos[0]-1),int(plotPos[1]-1)),
                    (int(plotPos[0]+1),int(plotPos[1]+1)),
                    color = (0, 255, 0),
                    thickness = 3)

    plotPos = (int(thymio.pos_est[0][0] /x_scale * 1000),int(thymio.pos_est[0][1] / y_scale * 1000))

    edt_frame = cv2.circle(frame_cropped,
                    plotPos,
                    radius = 7,
                    color = (0,255, 0),
                    thickness= 2)

    thymio.storePath(plotPos[0],plotPos[1])
    for i in range(1,len(thymio.xPlot)):
        plotPos = (int(thymio.xPlot[i]),int(thymio.yPlot[i]))
        edt_frame = cv2.circle(frame_cropped,
                    plotPos,
                    radius = 1,
                    color = (255,0 , 0),
                    thickness= 2)

    return edt_frame

with ClientAsync() as client:
    async def main():
        with await client.lock() as node:
            # Sampling time: 
            delta_t = 0.6
            # Place holder image 2 be replaced by a camera pic
            cap = cv2.VideoCapture(0)
            frame = cap.read()[1]

            plt.imshow(frame)
            plt.show()

            # Initialize the kalman filter:
              # Initial P0 is initialized as eye(5) and state as zero state
              # Would be nice to use as an initiale state first aruco kalman measurement
            #Q_kal, R_kal, P_kal = initialize_kalman()

            # Get initial vision for navigation 
            globalMap,goal,start,R = getNavigationMap(frame)
            frame_cropped = doPerspectiveTransform(frame,R)
            startPos = np.array([pose(frame_cropped)])
            startVel = np.zeros((1,2))
            thymio = myThymio(startPos ,startVel,initialize_kalman(),delta_t)

            j = 0
            while j <= 10:
                j += 1
                frame = cap.read()[1]
                frame_cropped = doPerspectiveTransform(frame,R)
                startPos = np.array([pose(frame_cropped)])
                if startPos[0][0] != -1:
                    break
            
            startVel = np.zeros((1,2))
            thymio = myThymio(startPos ,startVel,initialize_kalman(),delta_t)
            
            plt.imshow(frame_cropped)
            plt.scatter(goal[0],goal[1],marker ='x',linewidths=30)
            plt.scatter(start[0],start[1],marker = 'x',linewidths=30)
            plt.show()

            frame_height = len(frame_cropped)
            frame_width = len(frame_cropped[0])
            out = cv2.VideoWriter('thymioVid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))

            
            rx, ry, ryaw = gp.path_planning(globalMap, start, goal, thymio.delta_t, 150)
            i = 0
            t = time.time()

            thymio.savePath(rx,ry)
            
            pos = np.array([start[0],start[1],ryaw[0]]).T
            while True:
                print(i)
                print('thymio.pos_est: ',thymio.pos_est)
                print('thymio.vel_est: ',thymio.vel_est)
                await node.wait_for_variables({"prox.horizontal"})
                await node.wait_for_variables({"motor.left.speed"})
                await node.wait_for_variables({"motor.right.speed"})
                localObstacleAvoidance(node)

                frame = cap.read()[1]
                frame_cropped = doPerspectiveTransform(frame,R)
                thymio.camera_meas(pose(frame_cropped))
                kalman_measure(node,thymio)

                next_pos = np.array([rx[i],ry[i],ryaw[i]]).T
                print('next pos: ', next_pos)
                kal_pos = thymio.pos_est[0]
                kal_pos[2] = rm.angle2stdrange(pos[2])
                print('measured pos: ', kal_pos)
                

                edt_frame = makeVideo(thymio,frame_cropped)

                out.write(edt_frame)

                motor_right, motor_left = gp.compute_input(pos, next_pos, thymio.delta_t, 0.0002735)
                print(motor_right, motor_left)
                
                await node.set_variables({"motor.left.target":[motor_right],
                        "motor.right.target":[motor_left]})
    
                t+=thymio.delta_t
                await client.sleep(max(0,t-time.time()))
                print('time: ',time.time())
             
                pos = next_pos
                i += 1
                if i >= len(rx)-1:
                    await node.set_variables({"motor.left.target":[0],
                        "motor.right.target":[0]})
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    client.run_async_program(main)
