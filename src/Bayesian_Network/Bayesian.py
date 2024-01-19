import os
import cv2
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm
import pymc as pm
import tensorflow as tf

from EKF.data_preprocessing import (
    load_camera_image, load_lidar_data, read_kitti_calibration, apply_calibration,
    synchronize_data
)
from EKF.sensor_fusion import (
    object_detection_function, save_processed_data
)
from yolov7.models.experimental import attempt_load
from yolov7.utils.plots import plot_one_box

tfd = tfp.distributions

"""
This Bayesian script is used to fuse the data from the LiDAR and the camera using the Bayesian Network.
In the mean time we also detect objects in the camera image using YOLOv7. 
It also calculates the distance of the detected objects using the LiDAR data.
"""


# This function filters the LiDAR points based on the bounding box of the detected object
def filter_lidar_points(lidar_data, bbox, T_velo_cam, P_cam):
    # List to store filtered LiDAR points
    filtered_lidar_points = []

    P_cam = P_cam.reshape((3, 4))

    for point in lidar_data:
        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(point[:3], 1)

        # Transform point from LiDAR frame to camera frame
        transformed_point = np.dot(T_velo_cam, point_homogeneous)

        # Project points onto the camera image plane
        projected_point = np.dot(P_cam, transformed_point)

        # Normalize projected point
        projected_point = projected_point[:2] / projected_point[2]

        # Check if it's within the bbox
        x_in_range = bbox[0] <= projected_point[0] <= bbox[2]
        y_in_range = bbox[1] <= projected_point[1] <= bbox[3]

        if x_in_range and y_in_range:
            filtered_lidar_points.append(point)

    return filtered_lidar_points


# Calculates the avg distance to the detected object
def bay_calculate_distance_to_object(filtered_lidar_points, bbox):
    try:
        # Extract the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate distances for all filtered LiDAR points
        distances = [np.linalg.norm(point[:3]) for point in filtered_lidar_points]

        # only if there are enough points to compute a meaningful average
        if len(distances) > 3:
            distance = np.median(distances)
        else:
            # fall back to simple mean
            distance = np.mean(distances) if distances else None

        return distance

    except Exception as e:
        print(f"Error in bay_calculate_distance_to_object: {e}")
        return None


# Associate LiDAR points with corresponding pixels in the camera image
def associate_lidar_camera_data(projected_lidar, camera_image, T_velo_cam, P_cam):
    # Transform lidar points into camera coordinates
    projected_lidar, _ = apply_calibration(projected_lidar, T_velo_cam, P_cam)
    associated_data = []

    for point in projected_lidar:
        # Extract x and y coordinates
        x = int(point[0])
        y = int(point[1])

        # Check if the point is within the bbox
        is_x_in_bounds = 0 <= x < camera_image.shape[1]
        is_y_in_bounds = 0 <= y < camera_image.shape[0]

        if is_x_in_bounds and is_y_in_bounds:
            associated_data.append((point, camera_image[y, x]))

    return associated_data


def visualize_lidar_points_on_image(camera_image, lidar_points, bbox, T_velo_cam, P_cam):
    P_cam = P_cam.reshape((3, 4))

    lidar_points = np.asarray(lidar_points)
    points_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))

    # Perform the dots
    points_camera_frame = np.dot(points_homogeneous, T_velo_cam.T)
    points_image_plane = np.dot(points_camera_frame, P_cam.T)

    # Normalize the points by the third (z) coordinate
    points_image_plane /= points_image_plane[:, 2, np.newaxis]

    # Iterate through the points and draw them if they are within the bbox
    for i, point in enumerate(points_image_plane):
        x = int(point[0])
        y = int(point[1])

        x_in_range = bbox[0] <= x <= bbox[2]
        y_in_range = bbox[1] <= y <= bbox[3]

        if x_in_range and y_in_range:
            cv2.circle(camera_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    return camera_image


def estimate_distance_with_bayesian_model(lidar_data, detected_object_bbox, T_velo_cam, P_cam):
    with pm.Model() as model:
        # Define the prior for the distance
        distance_mean = pm.Normal('distance_mean', mu=15, sigma=30)

        # Use the filter_lidar_points function to get relevant LiDAR points
        filtered_lidar_points = filter_lidar_points(lidar_data, detected_object_bbox, T_velo_cam, P_cam)

        # Calculate the average distance to the detected object
        observed_distance = bay_calculate_distance_to_object(filtered_lidar_points, detected_object_bbox)

        trace = None

        if observed_distance is not None:
            # Define the likelihood of the observed data given the model
            pm.Normal('obs', mu=distance_mean, sigma=5, observed=observed_distance)

            # Sampling from the posterior
            trace = pm.sample(1000, return_inferencedata=True)

        if trace is not None:
            # Extract the distance estimate from the posterior
            distance_estimate = np.mean(trace.posterior['distance_mean'].values)
            return distance_estimate
        else:
            return None  # or any other appropriate handling


# Main function
def main():
    # Use CPU if GPU is not available!!
    model = attempt_load('yolov7.pt', map_location='cuda')
    calib_file_path = '../data/calib/calib.txt'
    camera_dir = '../data/image_2_1'
    lidar_dir = '../data/velodyne_1'
    output_dir = '../data/Bayesian_processed'
    timestamp_file = '../data/timestamp/times.txt'

    # Read calib file
    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)

    # Synchronize data based on timestamps
    data_pairs = synchronize_data(camera_dir, lidar_dir, timestamp_file)

    # Loop through each pair of data and fuse them using Bayesian network
    for image_path, lidar_path, timestamp in tqdm(data_pairs, desc='Processing frames'):
        camera_image = load_camera_image(image_path)
        lidar_points = load_lidar_data(lidar_path)

        projected_points, _ = apply_calibration(
            lidar_points,
            T_velo_cam,
            P_cam
        )

        # Detect objects in the camera frame and extract features
        camera_image_with_detections, detected_objects = object_detection_function(
            camera_image,
            model,
            output_dir
        )

        # Process each detected object
        for detected_object in detected_objects:
            bbox = detected_object['bbox']
            label = detected_object['label']

            distance_estimate = estimate_distance_with_bayesian_model(
                lidar_points,
                bbox,
                T_velo_cam,
                P_cam
            )

            # Visualize the bounding box and distance on the image
            if distance_estimate is not None:
                label_with_distance = f'{label} | Dist: {distance_estimate:.2f}m'
                plot_one_box(bbox, camera_image_with_detections, label=label_with_distance, color=(0, 255, 0),
                             line_thickness=1)

            # Visualize the LiDAR points on the image
            camera_image_with_detections = visualize_lidar_points_on_image(
                camera_image_with_detections,
                projected_points,
                bbox,
                T_velo_cam,
                P_cam
            )

        # Save processed data
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_processed_data(camera_image_with_detections, lidar_points, output_dir, base_filename)


if __name__ == "__main__":
    main()
