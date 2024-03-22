# README
There may be a lot more comments than preferred but this was done with the intention of passing this code down to Brunel
Autonomous Racing Team in order to help future team members working on sensor fusion ;)

# Download data 
To download data you need to register for KITTI dataset and download the data from the following link: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Download odometry data set color 65GB, velodyne laser data 80GB and calibration files 1mb.

After that has been downloaded select any sequence however I would recommend sequence 00 since the code has been adjusted for that specific sequence calibration file.
Other sequences will require the calibration pre-processing to be adjusted accordingly.

# How to run
1. You can run by using notebook BayesianNetwork and EKF if you do not have the GPU power and correct packages installed
on local PC.
2. I made the notebook to run on Google Colab. You can run the notebook on Google Colab by uploading the data to Google Drive.
3. The Google Drive requires the folders to be in the following structure:
    - calib
    - image2
    - processed (or bayesian processed just)
    - timestamp
    - velodyne
If the names dont match just scroll to the bottom of code and change the names accordingly.

Have notebook or local environment set up? 
Firstly after installing yolov7 go to models - experimental and change the code to the one in this zip file. Same goes
for detect.py since they both have been modified and WON'T work with the original code. 

Then restart the session for changes to happen and run the code. Enjoy. 

any issues please contact me via github or email. 