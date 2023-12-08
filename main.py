import numpy as np
import cv2
from matplotlib import pyplot as plt
from myThymio import myThymio
from tdmclient import ClientAsync
import sys
import os
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'Vision'))
sys.path.insert(0, os.path.join(os.getcwd(), 'Video'))
sys.path.insert(0, os.path.join(os.getcwd(), 'GlobalNavTest'))
sys.path.insert(0, os.path.join(os.getcwd(), 'Path_planning'))

from Vision.visionMain import getNavigationMap
from Vision.visionMain import doPerspectiveTransform
from Vision.camera_pose import pose

import Path_Planning.global_path as gp

from ekf import *
from local_obst_avoidance import *
from Video.mainVideo import *
from motion import *


with ClientAsync() as client:
    async def main():
        with await client.lock() as node:
            # Sampling time: 
            delta_t = 0.2
            # Initialize camera object
            cap = cv2.VideoCapture(0)
            fig = plt.figure()
            if (cap.isOpened() == False):
                print("Unable to read camera feed")
            frame = cap.read()[1]
            plt.imshow(frame)
            plt.show()

            # Get initial vision for navigation 
            globalMap,goal,start,R = getNavigationMap(frame)
            frame_cropped = doPerspectiveTransform(frame,R)
            startPos = np.array([pose(frame_cropped)])
            startVel = np.zeros((1,2))
            thymio = myThymio(startPos ,startVel,initialize_kalman(),delta_t)

            # Initialize object for video capture
            frame_height = len(frame_cropped)
            frame_width = len(frame_cropped[0])
            out = cv2.VideoWriter('thymioVid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))

            # Get initial vision for navigation
            globalMap,goal,start,R = getNavigationMap(frame)

            # Get and store path planning
            rx, ry, ryaw = gp.path_planning(globalMap, start, goal, thymio.delta_t, 50)
            thymio.savePath(rx,ry)
            N_points = len(rx)

            # Run program until Thyimo is sufficiently close to the goal
            i = 0
            while True:
                print('i',i)
                print('Pos est:',thymio.pos_est)
                print('Meas:',thymio.pos_meas)
                t_start = time.time()
                
                # Wait for variables 
                await node.wait_for_variables({"prox.horizontal"})
                await node.wait_for_variables({"motor.left.speed"})
                await node.wait_for_variables({"motor.right.speed"})

                # Read measurements and do kalman filtering
                frame = cap.read()[1]
                frame_cropped = doPerspectiveTransform(frame,R)
                thymio.camera_meas(pose(frame_cropped))
                
                # Save to video
                edt_frame = makeVideo(thymio,frame_cropped)
                out.write(edt_frame)

                
                # Get kalman filter pose update
                kalman_measure(node,thymio)

                # Check sensors
                obstacle_detect(node,thymio)
                thymio.minDist2path()

                # Choose appropriate navigation
                if thymio.obstacleFound or thymio.avoidingObstacle:
                    localObstacleAvoidance(thymio)
                else:
                    print('Following global path')
                    doControl(thymio)

                # Increment sample time with compilation time to ensure constant sampling frequency
                t_end = time.time()
                t_delta = t_end-t_start
                
                await node.set_variables({"motor.left.target":[thymio.motorLeft],
                        "motor.right.target":[thymio.motorRight]})
                await client.sleep(thymio.delta_t-t_delta)

                i += 1
                # Exit condition
                if thymio.current_point >= N_points-1:
                    await node.set_variables({"motor.left.target":[0],
                        "motor.right.target":[0]})
                    break
                cv2.imwrite("out.jpg", edt_frame)
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    client.run_async_program(main)
