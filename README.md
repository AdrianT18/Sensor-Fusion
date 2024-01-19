# Lidar-Camera Fusion for Enhanced Object and Distance Detection as part of FS-AI
P.S. New repo as there where issues with previous one. 
## Overview

This project is part of my dissertation for Brunel University, in collaboration with the Brunel Autonomous Racing team.
We are prepending up for the FS-AI competition in July 2024. My dissertation focuses on Multi-Sensor Fusion,
specifically the fusion of a LiDAR sensor with a Stereo Camera. The aim is to improve object detection and distance
measurement, thereby enhancing the performance of various submodules within the autonomous racing vehicle.

## Objectives

- **Enhanced Object Detection**: By fusing data from LiDAR and Stereo Cameras, the system aims to achieve more accurate
  and reliable object detection.
- **Precise Distance Measurement**: Improve the accuracy of distance detection to facilitate better navigation and
  maneuvering.
- **Support Submodules**: Provide enriched data to other submodules, ultimately contributing to increased speed and
  accuracy of the autonomous vehicle.

## Methodology

### Sensor Fusion

- **LiDAR and Stereo Camera Integration**: Combining the depth information from LiDAR with the visual data from Stereo
  Cameras.

### Algorithms

1. **Extended Kalman Filter (EKF)**: The EKF is used for combining sensor data in a statistically optimal way. It's an
   advanced version of the Kalman Filter, capable of handling non-linear system models, making it well-suited for
   complex environments in autonomous racing.
2. **Bayesian Network**: This network is utilized for its exceptional ability to model the probabilistic relationships
   between various sensor inputs, specifically between LiDAR and Stereo Cameras. Its strength lies in handling the
   inherent uncertainties and interdependencies, making it particularly effective for complex tasks like object
   detection and distance estimation in dynamic environments, as encountered in autonomous racing.
3. **[Algorithm 3]**

### Evaluation

- Each algorithm will be rigorously tested and evaluated based on its contribution to object detection accuracy and
  distance measurement precision.

## Current Progress

- Implemented the EKF algorithm for sensor fusion.
- Added object detection using yolov7 to the EKF algorithm.
- Added distance measurement using Lidar to the EKF algorithm.
- Successfully Fused the data from the two sensors.
- Added Bayesian Network for probabilistic modelling of the sensor data.
- Also integrated yolov7 in Bayesian Network and distance detection.

## To Do

- [ ] Complete the Bayesian Network implementation.
- [ ] Test and evaluate the Bayesian Network.
- [ ] Use the better performing algorithm with real-time LiDAR and Stereo Camera data from the autonomous vehicle
  provided by Brunel Autonomous Racing.

## Current demo
### Extended Kalman Filter
This small video is the fused sensors in a fram -  We can see that it has been succesfully done as the detected object distance is being calculated using the LiDAR points. In addition to this the overlay of the points. They are there for fun right now. 


https://github.com/AdrianT18/Sensor-Fusion/assets/100729061/56632bc4-e02f-4ea8-bae4-51f55de1e6ba

### Bayesian Network
This is to just show that Bayesian network is much better then EKF. More accurate when it come to distance detection. EKF was detecting the right car to be 14.45m.
![image](https://github.com/AdrianT18/Sensor-Fusion/assets/100729061/a3bfa8b2-1ae4-4d87-b525-a7db3759fbc7)

## Installation and Usage

```bash
# Instructions for setting up the project
Coming soon...
```

