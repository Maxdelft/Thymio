# Thymio Project
By: Iskandar Khemakhem, Erik Börve, Maximilian van Amerongen, Romane Belda

Basics of Mobil Robotics, EPFL, 12-12-2021


## Abstract
This report presents the choices we made to achive the project goal of navigating the differential drive robot, refered to as "Thymio", in an environment containing known and unknown obstacles. Furthermore, certain parts of the code will be highlighted to support the made design choices.

To achieve the goal, different image processing techniques have been applied to sense Thymio's enviornment and to build up a map. Different path planning techniques were applied to the map to come up with a path that navigates the robot from its start to its goal position.
A Kalman Filter was designed to fuse different sensors and localize the Thymio on the map.

Finally, simple motion control combined with local obstacle avoidance has been applied to achieve path following and to avoid unknown obstacles.

The resulting Motion control is displayed in the video below (significantly speed up). The corresponding video in real time and high resolution can be found in the "Video" folder.


https://user-images.githubusercontent.com/81572776/153708500-784dddef-6b9c-4efc-996d-09a122e975ed.mp4


The project is described in more detail in the report.
