import global_path as gp
import cv2

def makeVideo(thymio,frame_cropped):
    '''
    Plots path visuals in transformed frame
    '''

    # Plot optinal path
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

    # Plot current position of thymio
    plotPos = (int(thymio.pos_est[0][0] /x_scale * 1000),int(thymio.pos_est[0][1] / y_scale * 1000))
    edt_frame = cv2.circle(frame_cropped,
                    plotPos,
                    radius = 7,
                    color = (0,255, 0),
                    thickness= 2)

    # Plot all past position of the thymio
    thymio.storePath(plotPos[0],plotPos[1])
    for i in range(1,len(thymio.xPlot)):
        plotPos = (int(thymio.xPlot[i]),int(thymio.yPlot[i]))
        edt_frame = cv2.circle(frame_cropped,
                    plotPos,
                    radius = 1,
                    color = (255,0 , 0),
                    thickness= 2)

    return edt_frame