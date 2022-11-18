# PerilScope Source Code
This project is the source code for PerilScope, a monitoring system for police vehicles that sends smart alerts for objects and events detected around the vehicle. This placed 1st overall at Aggies Invent: First Responders, the premier 48-hour design contest at Texas A&M open to all undergraduates and grad students.

## Performance


# Approach
This project runs on a Jetson Nano with an attached camera and lidar sensor. It runs the lightweight Yolo v5 model to locate and classify images from the camera sensor, and it uses the lidar sensor to measure the object's distance and velocity relative to the vehicle. With this information, my program determines whether the objects and their distance/velocity pose a threat to an officer. If so, it sends an alert 

# Credit
This project utilizes code from https://github.com/amirhosseinh77/JetsonYolo to set up the Yolo model and interact with the Jetson Nano camera. Everything else was written by me.
