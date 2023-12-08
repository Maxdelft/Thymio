import numpy as np
import math
import matplotlib.pyplot as plt
import Path_planning.robot_model as rm
from Path_planning.Voronoi_path import VoronoiRoadMapPlanner

from scipy import interpolate
#from Path_planning.A_star_path import AStarPlanner


def get_scale_factor_to_reality(map):
    #computes the scaling factor to the map array compared to real map measurments in mm (891mm*1260mm)
    # inputs: map array
    # outputs: scaling factor in the x- and y-directions.
    map_factor_x = 891/np.shape(map)[0]
    map_factor_y = 1260/np.shape(map)[1]
    return map_factor_x, map_factor_y

def scale_map(map, factor_x, factor_y, sx, sy, gx, gy, robot_radius, grid_size, MAX_EDGE_LEN):
    # scale a given map by a given factor using a maximum mask. The start and goal positions 
    # and the robot and grid sizes and the maximum edge length (for the different path planning algorithm) are also scaled accordingly.
    # input: map: array
    #        sx, sy: start position in mm
    #        gx, gz: end position in mm
    #        robot_radius, grid_siye, MAX_EDGE_LEN: measurments in mm
    # output map, start and end positions and measurments scaled down.
    map_factor_x, map_factor_y = get_scale_factor_to_reality(map)
    i = 0
    j = 0
    map_scaled = []

    while (i+factor_y<=np.shape(map)[0]):
        while (j+factor_x<=np.shape(map)[1]):
            box = map[i:i+factor_x,j:j+factor_y]
            max = np.min(box)
            map_scaled.append(max)
            j = j+factor_x
        i = i+factor_y
        j = 0
    map_scaled=np.reshape(map_scaled,(int(np.shape(map)[0]/factor_x),int(np.shape(map)[1]/factor_y)))   
    # start and goal position
    sx = sx/(map_factor_x*factor_x)
    sy = sy/(map_factor_y*factor_x)
    gx = gx/(map_factor_x*factor_x)
    gy = gy/(map_factor_y*factor_x)
    
    robot_radius = np.ceil(robot_radius/(((map_factor_x+map_factor_y)/2)*((factor_x+factor_y)/2)))
    grid_size = np.ceil(grid_size/(((map_factor_x+map_factor_y)/2)*((factor_x+factor_y)/2)))
    MAX_EDGE_LEN = MAX_EDGE_LEN/(((map_factor_x+map_factor_y)/2)*((factor_x+factor_y)/2))
    
    return [map_scaled, sx, sy, gx, gy, robot_radius, grid_size ,MAX_EDGE_LEN]

def create_map(map_scaled):
    # extract the x and y coordinates of the obstacles and sets the borders of the map.
    # input: a map array
    # output: x and y coordinates of the obstacles including the border
    # set obstacle positions
    ox, oy = [], []
    for i in range(np.shape(map_scaled)[0]):
        for j in range(np.shape(map_scaled)[1]):
            if map_scaled[i,j] == 0:
                ox.append(j)
                oy.append(i)
    for i in range(0, np.shape(map_scaled)[1]):
        ox.append(i)
        oy.append(0)
    for i in range(0, np.shape(map_scaled)[1]):
        ox.append(i)
        oy.append(np.shape(map_scaled)[0])
    for i in range(0, np.shape(map_scaled)[0]):
        ox.append(0)
        oy.append(i)
    for i in range(0, np.shape(map_scaled)[0]):
        ox.append(np.shape(map_scaled)[1])
        oy.append(i)
    return ox, oy

def plot_path(ox, oy, rx, ry, sx, sy, gx, gy):
    # plots a map and a path on the map. Start and end positions are marked differently.
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")   
    plt.plot(rx,ry, ".r")
    plt.show()

def scale_path(rx,ry, map, factor_x, factor_y):
    # scales a given path to meter
    # input: rx, ry: lists of x and y coordinates of a path
    #        map: array containing the map 
    #        factor_x, factor_y: scaled factors used on the map previously
    # output: rx_sclaed, ry_scaled: scaled x- and y-directions of the path
    rx_scaled = []
    ry_scaled = []
    map_factor_x, map_factor_y = get_scale_factor_to_reality(map)
    for i in range(len(rx)):
        rx_scaled.append((rx[i]*(map_factor_x*factor_x))/1000)
        ry_scaled.append((ry[i]*(map_factor_y*factor_y))/1000)
    return rx_scaled, ry_scaled

def compute_rotation(rx,ry,gyaw):
    # computes for a given (rx,ry)-path the angle needed by the robot.
    # input: rx, ry: list of path coordinates
    # output: ryaw: list of the oath angles.
    ryaw = []
    for i in range(np.shape(rx)[0]-1):
        angle = np.arctan2(ry[i+1]-ry[i],rx[i+1]-rx[i])
        ryaw.append(angle)
    last_element = ryaw[-1] 
    ryaw.append(last_element)
    return ryaw

def compute_input(pos, goal, dt, factor):
    # computes the input necessary to go from one position to the goal using inverse kinematics 
    # of the differentilal drive robot. 
    # input: pos, goal: arrays with the start and goal poses.
    #        dt: time step 
    #        factor: computed factor to tranform to wheel speed
    # output: motor_right, motor_left: thymio wheel speeds as integer
    R = 0.022  
    L = 0.095 
    v, w = rm.inverse_kinematics(goal,pos,dt)
    motor_right, motor_left = rm.motor_speed(v,w,R,L,factor)
    return int(motor_right), int(motor_left)
    
def compute_open_loop_inputs(rx_scaled,ry_scaled,ryaw,factor, dt, x0):
    # computes the input in an open-loop dirve.
    # input: rx_scaled, ry_scaled, ryaw: lists of the path positions and orientation
    #        dt: time step size
    #        x0: array of the start position
    R = 0.022  
    L = 0.095   
    x_old = x0
    motor_inputs = np.array([[0],[0]]).T
    robot_inputs = np.array([[0],[0]]).T
    for i in range(np.shape(rx_scaled)[0]):
        x = np.array([rx_scaled[i],ry_scaled[i],ryaw[i]]).T
        v, w = rm.inverse_kinematics(x,x_old,dt)
        robot_inputs = np.append(robot_inputs, np.array([[v], [w]]).T,axis = 0)
        motor_right, motor_left = rm.motor_speed(v,w,R,L,factor)
        x_old = x
        motor_inputs = np.append(motor_inputs, np.array([[int(motor_right)], [int(motor_left)]]).T,axis = 0)
    return motor_inputs, robot_inputs

def rotate(position, goal, factor, dt):
    T = 3
    N = T/dt
    angle = np.linspace(position[2],goal[2],N).tolist()
    turn_x = []
    turn_y = []
    turn_yaw = []

    for i in range(N):
        turn_x.insert(0,position[0])
        turn_y.insert(0,position[1])
        turn_yaw.insert(0,angle[N-i-1])
    motor_inputs, = compute_open_loop_inputs(turn_x, turn_y, turn_yaw, factor, dt, position)
    return motor_inputs

def flip_cooridinate(x1):

    return np.array([x1[0],-(x1[1]-0.891),rm.angle2stdrange(x1[2]-(np.pi/2))]).T

def path_planning(map, startPos, goalPos, dt, num):
    # global method that generates a path scaled in m from a given map amd a given start and end positions.
    factor_x = 5
    factor_y = 5

    map_factor_x, map_factor_y = get_scale_factor_to_reality(map)
    print('start : ',startPos)
    print('goal :', goalPos)
    sx = startPos[0]*map_factor_x  #startposition in mm real scate
    sy = startPos[1]*map_factor_y
    gx = goalPos[0]*map_factor_x
    gy = goalPos[1]*map_factor_y

    # define syaw depending on the results
    syaw = -math.pi/2
    x0 = np.array([sx/1000,sy/1000,syaw])
    print('x0= ', x0)
    # define gyaw depensing on the results
    gyaw = math.pi/2
    xend = np.array([gx/1000,gy/1000,gyaw])
    print('xend= ', xend)

    factor = 0.0002735
    grid_size = 50       # [mm] for the A*
    MAX_EDGE_LEN = 50  # [mm]  for prm and voronoi
    robot_radius = 70  # [mm] 

    [map_scaled, sx, sy, gx, gy, robot_radius, grid_size, MAX_EDGE_LEN] = scale_map(map, factor_x, factor_y, sx, sy, gx, gy, robot_radius, grid_size, MAX_EDGE_LEN)
    [ox, oy] = create_map(map_scaled)

    #Voronoi algorithm
    N_KNN = 1500  # number of edges from one sampled point
    MAX_EDGE_LEN = 15
    voronoi = VoronoiRoadMapPlanner(N_KNN, MAX_EDGE_LEN)
    rx, ry = voronoi.planning(sx, sy, gx, gy, ox, oy, robot_radius)   
    
    print(np.shape(rx),np.shape(ry))

    smooth_func, u= interpolate.splprep([rx, ry])
    rxint, ryint = interpolate.splev(np.linspace(0, 1, num), smooth_func)

    rx_scaled, ry_scaled = scale_path(rxint, ryint, map_scaled , 1, 1)
    print(np.shape(rx_scaled),np.shape(ry_scaled))

    ryaw = compute_rotation(rx_scaled,ry_scaled,gyaw)
    N = int(1/dt)

    print(np.shape(rx_scaled),np.shape(ry_scaled))

    return rx_scaled, ry_scaled, ryaw

def get_map(map, startPos, goalPos):
    #method for the report
    factor_x = 3
    factor_y = 3

    map_factor_x, map_factor_y = get_scale_factor_to_reality(map)
    sx = startPos[0]*map_factor_x  #startposition in mm real scate
    sy = startPos[1]*map_factor_y
    gx = goalPos[0]*map_factor_x
    gy = goalPos[1]*map_factor_y

    grid_size = 50       # [mm] for the A*
    MAX_EDGE_LEN = 50  # [mm]  for prm and voronoi

    robot_radius = 80  # [mm] 

    [map_scaled, sx, sy, gx, gy, robot_radius, grid_size, MAX_EDGE_LEN] = scale_map(map, factor_x, factor_y, sx, sy, gx, gy, robot_radius, grid_size, MAX_EDGE_LEN)
    print('resolution of the scaled map: ',np.shape(map_scaled))
    print('robot radius:', robot_radius)
    print('grid size:', grid_size)

    print('MAX_EDGE_LEN:', MAX_EDGE_LEN)

    [ox, oy] = create_map(map_scaled)
    return ox, oy, sx, sy, gx, gy, robot_radius, grid_size
    


