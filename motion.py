import numpy as np

def moveThymio(thymio):
    '''
    Sets the wheel speed target in the thymio object
    '''
    move = 80
    thymio.setSpeed(move,move)

def rotateThymio(thymio):
    '''
    Rotates thymio towards angle between current and following point in global path.
    Based on extended kalman filter pose estimate.
    '''
    rot = 20
    if thymio.angle <= np.pi:
        if thymio.measAngle > thymio.angle + np.pi or thymio.measAngle < thymio.angle:
            thymio.setSpeed(-rot,rot)
        else:
            thymio.setSpeed(rot,-rot)
    if thymio.angle > np.pi:
        if thymio.measAngle > thymio.angle - np.pi and thymio.measAngle < thymio.angle:
            thymio.setSpeed(-rot,rot)
        else:
            thymio.setSpeed(rot,-rot)

def doControl(thymio):
    '''
    Swaps between moving forward and rotating towards following point in path.
    Based on difference in distance and angle
    '''
    thymio.distance2path()
    thymio.angle2point()

    print('Angle of next point: ',thymio.angle)
    print('Estimated angle: ',thymio.measAngle)
    print('Point distance:',thymio.distPoint)

    if abs(thymio.distPoint) < 0.03:        
        rotateThymio(thymio)

        if abs(thymio.anglePoint) < 0.1 or abs(thymio.anglePoint) > 2*np.pi-0.1:
            thymio.setSpeed(0,0)
            thymio.updatePoint()
    else:
        moveThymio(thymio)
