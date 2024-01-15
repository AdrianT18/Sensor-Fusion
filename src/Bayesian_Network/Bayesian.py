import os
import cv2
import numpy as np
from numpy.random import multivariate_normal
from tqdm import tqdm
from EKF.data_preprocessing import (
    load_camera_image, load_lidar_data, read_kitti_calibration, apply_calibration,
    synchronize_data
)
from EKF.sensor_fusion import (
    object_detection_function, calculate_distance_to_object, save_processed_data
)
from yolov7.models.experimental import attempt_load
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm

tfd = tfp.distributions

"""
This Bayesian script is used to fuse the data from the LiDAR and the camera using the Bayesian Network.
In the mean time we also detect objects in the camera image using YOLOv7. 
It also calculates the distance of the detected objects using the LiDAR data.
"""


# def extract_lidar_data_for_objects(calibrated_lidar_points, detected_objects, projected_points):
#     lidar_data_for_objects = []
#     for detected_object in detected_objects:
#         bbox = detected_object['bbox']
#         label = detected_object.get('label', 'Unknown')
#         print(projected_points.shape)
#         distance = calculate_distance_to_object(calibrated_lidar_points, projected_points, bbox)
#         if distance is not None:
#             label_with_distance = f'{label} | Dist: {distance:.2f}m'
#             plot_one_box(bbox, camera_image_with_detections, label=label_with_distance, color=(0, 255, 0),
#                          line_thickness=1)
#
#     return lidar_data_for_objects


# This function filters the LiDAR points based on the bounding box of the detected object
def filter_lidar_points(lidar_data, bbox, calib_file_path, distance):
    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)
    # List to store filtered LiDAR points
    filtered_lidar_points = []

    P_cam = P_cam.reshape((3, 4))

    # print(f"Initial LiDAR points count: {len(lidar_data)}")

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
        if bbox[0] <= projected_point[0] <= bbox[2] and bbox[1] <= projected_point[1] <= bbox[3]:
            filtered_lidar_points.append(point)
    # print(f"Filtered LiDAR points count: {len(filtered_lidar_points)}")

    return filtered_lidar_points


# Visualizes the fusion of the LiDAR and the camera
def visualize_fusion(camera_image, detected_objects, projected_points, lidar_data_for_objects):
    for detected_object, lidar_data in zip(detected_objects, lidar_data_for_objects):
        # Extract bbox, label, and confidence
        bbox = detected_object['bbox']
        label = detected_object.get('label', 'Unknown')
        confidence = detected_object.get('confidence', 0)

        # Draw bounding box and label
        cv2.rectangle(camera_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # Display label and confidence score
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(camera_image, label_text, (int(bbox[0]), max(int(bbox[1]) - 20, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 2)

        # Calculate distance to object
        distance = bay_calculate_distance_to_object(lidar_data, projected_points, bbox)
        if distance is not None:
            distance_text = f"Dist: {distance:.2f}m"
        else:
            distance_text = "Dist: N/A"
        cv2.putText(camera_image, distance_text, (int(bbox[0]), max(int(bbox[1]) - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 255, 0), 2)

        # Overlay LiDAR points that fall within the bounding box onto the camera image
        for point in projected_points:
            x, y = int(point[0]), int(point[1])
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                cv2.circle(camera_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    return camera_image


# Calculates the avg distance to the detected object
def bay_calculate_distance_to_object(lidar_data, projected_points, bbox):
    try:
        # coordinates of the bbox
        x1, y1, x2, y2 = bbox
        # print(f"Bounding Box: {bbox}")
        # Find the indices of the points that fall within the bbox
        associated_points_indices = [i for i, p in enumerate(projected_points) if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]

        # print(f"Initial LiDAR points count: {len(lidar_data)}")
        # print(f"Associated points count: {len(associated_points_indices)}")

        # Select the points that fall within the bbox
        associated_lidar_points = [lidar_data[i] for i in associated_points_indices if i < len(lidar_data)]

        # print(f"Filtered LiDAR points count: {len(associated_lidar_points)}")

        if not associated_lidar_points:
            # print("No valid LiDAR points after filtering")
            return None  # No valid LiDAR points after filtering

        # Calculate the average distance of the associated LiDAR points
        distances = [np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in associated_lidar_points]

        return np.mean(distances) if distances else None

    except Exception as e:
        # print(f"Error in calculate_distance_to_object: {e}")
        return None


# Associate LiDAR points with corresponding pixels in the camera image
def associate_lidar_camera_data(projected_lidar, camera_image, T_velo_cam, P_cam):
    # Transform lidar points into camera coordinates
    projected_lidar, _ = apply_calibration(projected_lidar, T_velo_cam, P_cam)
    associated_data = []

    for point in projected_lidar:
        # Extract x and y coordinates (2D)
        x, y = int(point[0]), int(point[1])

        # Check if the point is within the image boundaries
        if 0 <= x < camera_image.shape[1] and 0 <= y < camera_image.shape[0]:
            associated_data.append((point, camera_image[y, x]))

    return associated_data


# Define the prior distribution over the environment state
def define_prior_distribution():
    distribution_type = 'Gaussian'

    # Setting the parameters for the prior distribution
    # Assuming objects to be 10-30m away from the vehicle - so choose 20m as the mean
    # Assuming std to be 5m as I expect them to be fairly noisy
    if distribution_type == 'Gaussian':
        mean = 20
        std = 5
        distribution = tfd.Normal(loc=mean, scale=std)
    # Assuming min distance of obj to be 2m and max to be 50m.
    elif distribution_type == 'Uniform':
        min_distance = 2
        max_distance = 50
        distribution = tfd.Uniform(low=min_distance, high=max_distance)

    return distribution


# Calculating the likelihood of the current LiDAR observation given based of previous predicted distances
def calculate_lidar_likelihood(lidar_data, previous_predicted_distances, lidar_measurement_error):
    likelihood = 0.0

    # Loop through each point in the LiDAR data
    for i, point in enumerate(lidar_data):
        # Calculate the distance of the point using Euclidean
        actual_distance = np.sqrt(np.sum(point[:3] ** 2))
        # Get the corresponding predicted distance
        predicted_distance = previous_predicted_distances[i]
        # Calculate the probability density of the actual distance being observed
        probability_density = norm.pdf(actual_distance, loc=predicted_distance, scale=lidar_measurement_error)
        # Add the log of the probability density to the likelihood
        likelihood += np.log(probability_density)

    return likelihood


# function to calculate the likelihood of camera data.
def calculate_camera_likelihood(camera_data, predicted_state, feature_extraction_function, measurement_error_cov):
    likelihood = 0.0

    # Extract features from the camera data using the provided feature extraction function.
    camera_features = feature_extraction_function(camera_data)
    # Extract features from the predicted state
    predicted_features = feature_extraction_function(predicted_state)
    for i, cam_feature in enumerate(camera_features):
        # Corresponding predicted feature
        pred_feature = predicted_features[i]
        # Calculate the probability density of the observed feature given the predicted feature
        probability_density = multivariate_normal.pdf(cam_feature, mean=pred_feature, cov=measurement_error_cov)
        # Add the log of the probability density to the likelihood
        likelihood += np.log(probability_density)

    return likelihood


# Define the likelihood functions for camera and LiDAR observations
def calculate_combined_likelihood(lidar_data, camera_data, prior_distribution):
    pass


# Use Bayesian inference to compute the posterior distribution
def compute_posterior_distribution(prior, likelihoods):
    pass


# Main fusion function that integrates the steps
def fuse_data_bayesian(associated_data):
    prior = define_prior_distribution()
    likelihoods = define_likelihood_functions(associated_data['camera'], associated_data['lidar'])
    posterior = compute_posterior_distribution(prior, likelihoods)
    return posterior


# Visualize the results of the fused data
def visualize_fused_data(fused_data):
    pass


# Main function
def main():
    # Use CPU if GPU is not available!!
    model = attempt_load('yolov7.pt', map_location='cuda')
    camera_dir = '../data/image_2_1'
    lidar_dir = '../data/velodyne_1'
    timestamp_file = '../data/timestamp/times.txt'
    calib_file_path = '../data/calib/calib.txt'
    output_dir = '../data/Bayesian_processed'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read calib file
    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)

    # Synchronize data based on timestamps
    data_pairs = synchronize_data(camera_dir, lidar_dir, timestamp_file)

    # List to store detection confidences and LiDAR data for objects
    detection_confidences = []
    lidar_data_for_objects = []

    # Process each pair of data
    for image_path, lidar_path, timestamp in tqdm(data_pairs, desc='Processing frames'):
        camera_image = load_camera_image(image_path)
        lidar_points = load_lidar_data(lidar_path)
        projected_points, _ = apply_calibration(lidar_points, T_velo_cam, P_cam)

        # Perform object detection on camera images
        camera_image_with_detections, detected_objects = object_detection_function(camera_image, model, output_dir)
        # print("Number of detected objects:", len(detected_objects))
        # for obj in detected_objects:
        #     print("Detected object bbox:", obj['bbox'])
        # lidar_data_for_objects = extract_lidar_data_for_objects(calibrated_lidar_points, detected_objects, projected_points)

        # Process each detected object
        for detected_object in detected_objects:
            bbox = detected_object['bbox']
            label = detected_object.get('label', 'Unknown')
            distance = calculate_distance_to_object(lidar_points, projected_points, bbox)

            # print("Calculated distance:", distance)
            # Filter LiDAR points based on the bbox
            if distance is not None:
                associated_lidar_points = filter_lidar_points(lidar_points, bbox, calib_file_path, distance)
                lidar_data_for_objects.append((associated_lidar_points, bbox, projected_points))

        detection_confidences = np.array(detection_confidences)

        print("Size of lidar_data_for_objects:", len(lidar_data_for_objects))
        # Perform the Bayesian fusion to enhance the detection confidence
        fused_confidences = bayesian_fusion(detected_objects, lidar_data_for_objects)
        for obj, fused_confidence in zip(detected_objects, fused_confidences):
            obj['fused_confidence'] = fused_confidence

        # Visualize the fusion of the LiDAR and the camera
        camera_image_with_fusion = visualize_fusion(camera_image, detected_objects, lidar_points,
                                                    lidar_data_for_objects)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_processed_data(camera_image_with_fusion, lidar_points, output_dir, base_filename)


if __name__ == "__main__":
    main()
