def obstacle_detect(node,thymio):
    '''
    Updates obstacle boolean in thymio object based on proximity sensors
    input:
        node object
        thymio object
    '''
    prox = node.v.prox.horizontal[0:5]
    sensorSum = prox[2] + prox[0] + prox[4]
    if sensorSum > 0:
        thymio.obstacleFound = True
    else:
        thymio.obstacleFound = False

def localObstacleAvoidance(thymio):
    '''
    Returns motor target for local obstacle avoidance.
    Based on proximity sensors and number of sample since last seen obstacle.

    Input:
        thymio object
    '''
    if thymio.obstacleFound:
        thymio.setSpeed(-25,25)
        thymio.avoidingObstacle = True
    else:
        if thymio.obstacle_iter < 3:
            thymio.setSpeed(50,50)
            thymio.obstacle_iter += 1
        else:
            thymio.setSpeed(35,-35)
            thymio.obstacle_iter = 0

            if thymio.smallestDist < 0.025:
                        print('Going back to path!')
                        thymio.setSpeed(0,0)
                        thymio.avoidingObstacle = False
                        thymio.current_point = thymio.closestPoint    # PERHAPS EDIT!!
