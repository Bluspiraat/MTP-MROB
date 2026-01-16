# MTP-MROB
Main branch of my MTP-MROB
This repository focuses on the training of a segmentation and terrain classification algorithm based on traditional U-Net
architectures and state-of-the-art transformer architectures. This segmented image is then used to perform global 
path planning of a rover through outdoor terrain.
The second repository of my thesis https://github.com/Bluspiraat/MTP-ROS2 focuses on the local path planning of the rover
following the waypoint given by the global planner.

## Branches
This repository contains three main branches each with their own respective task, it is used for convenience in this case
and not strictly following the intentions of Git. The branches are:
- main: Contains the A* path planning implementation
- dataset-creation: Focuses on the subsetting and alignment of training data
- network-training: Focuses on the training of networks and this branch is mostly used on external servers with more powerful hardware.