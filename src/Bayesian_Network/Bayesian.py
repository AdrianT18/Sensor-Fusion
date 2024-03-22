import os

import cv2
import numpy as np
import pymc as pm
import tensorflow_probability as tfp
from tqdm import tqdm

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
    P_cam = P_cam.reshape((3, 4))

    # Convert point to homogeneous coordinates
    ones = np.ones((lidar_data.shape[0], 1))
    lidar_homogeneous = np.hstack((lidar_data[:, :3], ones))

    # Transform point from LiDAR frame to camera frame
    transformed_points = np.dot(T_velo_cam, lidar_homogeneous.T)

    # Project points onto the camera image plane
    projected_points = np.dot(P_cam, transformed_points)

    # Normalize projected point
    normalized_points = projected_points[:2] / projected_points[2]

    # Check if it's within the bbox
    x_in_range = (bbox[0] <= normalized_points[0]) & (normalized_points[0] <= bbox[2])
    y_in_range = (bbox[1] <= normalized_points[1]) & (normalized_points[1] <= bbox[3])

    filtered_lidar_points = lidar_data[x_in_range & y_in_range]

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
            distance = np.mean(distances) if distances else None

        return distance

    except Exception as e:
        print(f"Error in bay_calculate_distance_to_object: {e}")
        return None


# Associate LiDAR points with corresponding pixels in the camera image.
# Used to loop through each point now it's done in a batch.
def associate_lidar_camera_data(projected_lidar, camera_image, T_velo_cam, P_cam):
    # Transform lidar points into camera coordinates
    projected_lidar, _ = apply_calibration(projected_lidar, T_velo_cam, P_cam)

    # Extract x and y coordinates
    x_coords = projected_lidar[:, 0].astype(int)
    y_coords = projected_lidar[:, 1].astype(int)

    # Check if the point is within the bbox
    valid_x = (x_coords >= 0) & (x_coords < camera_image.shape[1])
    valid_y = (y_coords >= 0) & (y_coords < camera_image.shape[0])
    valid_points = valid_x & valid_y

    valid_projected_lidar = projected_lidar[valid_points]
    pixel_values = camera_image[y_coords[valid_points], x_coords[valid_points]]

    associated_data = np.column_stack((valid_projected_lidar, pixel_values))

    return associated_data


# Visualize the LiDAR points on the camera image. Number of points is limited to valid ones.
def visualize_lidar_points_on_image(camera_image, lidar_points, bbox, T_velo_cam, P_cam):
    P_cam = P_cam.reshape((3, 4))

    # Convert points to homogeneous coordinates
    lidar_points = np.asarray(lidar_points)
    points_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))

    # Perform the dots
    points_camera_frame = np.dot(points_homogeneous, T_velo_cam.T)
    points_image_plane = np.dot(points_camera_frame, P_cam.T)

    # Normalize the points by the third (z) coordinate
    points_image_plane /= points_image_plane[:, 2, np.newaxis]

    # Iterate through the points and draw them if they are within the bbox
    x_in_range = (points_image_plane[:, 0] >= bbox[0]) & (points_image_plane[:, 0] <= bbox[2])
    y_in_range = (points_image_plane[:, 1] >= bbox[1]) & (points_image_plane[:, 1] <= bbox[3])
    valid_points = x_in_range & y_in_range

    for point in points_image_plane[valid_points]:
        x, y = int(point[0]), int(point[1])
        cv2.circle(camera_image, (x, y), radius=2, color=(0, 192, 255), thickness=-1)

    return camera_image


# Estimate distances to objects using Bayesian model (PYMC)
def estimate_distances_with_bayesian_model(observed_distances):
    with pm.Model() as model:
        # Prior for each object's distance
        distance_means = pm.Normal('distance_means', mu=15, sigma=30, shape=len(observed_distances))

        # Likelihood for observed data. Done in a batch instead of looping
        pm.Normal('observations', mu=distance_means, sigma=3, observed=observed_distances)

        # Perform MCMC sampling. 500 samples per chain, 2 chains
        trace = pm.sample(500, chains=2, return_inferencedata=True, progressbar=True, cores=2)

        if trace is not None:
            # Extracting distance estimates for each detected object
            distance_estimates = np.mean(trace.posterior['distance_means'].values, axis=(0, 1))
            return distance_estimates
        else:
            return None


# Main function
def main():
    batch_size = 5
    # Use CPU if GPU is not available!!
    model = attempt_load('/content/yolov7.pt', map_location='cuda')
    camera_dir = '/content/drive/MyDrive/Google Colab/image_2_1'
    lidar_dir = '/content/drive/MyDrive/Google Colab/velodyne_1'
    timestamp_file = '/content/drive/MyDrive/Google Colab/timestamp/times.txt'
    calib_file_path = '/content/drive/MyDrive/Google Colab/calib/calib.txt'
    output_dir = '/content/drive/MyDrive/Google Colab/Bayesian_processed'

    # Read calib file
    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)

    # Synchronize data based on timestamps
    data_pairs = synchronize_data(camera_dir, lidar_dir, timestamp_file)

    batch = []

    for image_path, lidar_path, timestamp in tqdm(data_pairs, desc='Processing frames'):
        camera_image = load_camera_image(image_path)
        lidar_points = load_lidar_data(lidar_path)

        # Object detection on the current frame
        camera_image_with_detections, detected_objects = object_detection_function(
            camera_image,
            model,
            output_dir
        )

        # Lists to store data for Bayesian estimation
        all_observed_distances = []
        all_bboxes = []
        all_labels = []

        # Process each detected object
        for detected_object in detected_objects:
            bbox = detected_object['bbox']
            label = detected_object['label']
            lidar_points_for_object = filter_lidar_points(lidar_points, bbox, T_velo_cam, P_cam)
            observed_distance = bay_calculate_distance_to_object(lidar_points_for_object, bbox)

            if observed_distance is not None:
                all_observed_distances.append(observed_distance)
                all_bboxes.append(bbox)
                all_labels.append(label)

        # Estimate distances using Bayesian model
        distance_estimates = estimate_distances_with_bayesian_model(all_observed_distances)

        # Visualize and label each object in the frame
        for i, (bbox, label, distance_estimate) in enumerate(zip(all_bboxes, all_labels, distance_estimates)):
            if distance_estimate is not None:
                label_with_distance = f'{label} | Dist: {distance_estimate:.2f}m'
                plot_one_box(bbox, camera_image_with_detections, label=label_with_distance, color=(0, 255, 0),
                             line_thickness=1)

            lidar_points_for_object = filter_lidar_points(lidar_points, bbox, T_velo_cam, P_cam)
            camera_image_with_detections = visualize_lidar_points_on_image(
                camera_image_with_detections,
                lidar_points_for_object,
                bbox,
                T_velo_cam,
                P_cam
            )

        # Save the processed frame in batches of 5
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        processed_frames = (camera_image_with_detections, lidar_points, output_dir, base_filename)
        batch.append(processed_frames)

        if len(batch) == batch_size or i == len(data_pairs) - 1:
            for data in batch:
                save_processed_data(*data)
            batch.clear()


if __name__ == "__main__":
    main()
