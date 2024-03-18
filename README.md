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

- [x] Complete the Bayesian Network implementation.
- [x] Test and evaluate the Bayesian Network.
- [ ] Use the better performing algorithm with real-time LiDAR and Stereo Camera data from the autonomous vehicle
  provided by Brunel Autonomous Racing.

## Current demo
[![Extended Kalman Filter and Bayasian Netwon demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]([https://www.youtube.com/watch?v=EkJVz7YGFek)


## Installation and Usage

```bash
# Instructions for setting up the project
1. Register for KITTI dataset
2. Download odometry data set color 65GB, velodyne laser data 80GB and calibration files 1mb.
3. Change the direcotry at the bottom of EKF and Bayesian Network.
    calib_file_path = '../data/calib/calib.txt'
    image_dir = '../data/image_2_1'
    lidar_dir = '../data/velodyne_1'
    calib_dir = '../data/calib'
    output_dir = '../data/processed'
    timestamp_file = '../data/timestamp/times.txt'
3.A - I would recommend this file structure: Ensure to use the correct sequence. 
    Data
        - calib
        - image2 ( the folder name from color )
        - processed ( where the processed images and velodyne points go to )
        - timestamp
        - velodyne
```

