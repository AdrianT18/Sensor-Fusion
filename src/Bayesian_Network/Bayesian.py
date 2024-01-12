import os
import cv2
import numpy as np
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


# Bayesian Fusion Function
# It fuses the data and using Bayesian interference to estimate the accuracy of object detection.
# Uses TensorFlow Probability in order to perform probabilistic modeling and Markov Chain Monte Carlo sampling.
def bayesian_fusion(detected_objects, lidar_data_for_objects):
    # Get the detection confidences from the detected objects
    detection_confidences = np.array([obj.get('confidence', 0) for obj in detected_objects])

    # Store distances
    lidar_distances = []

    # Calculate the distances for each detected object using points
    for obj_lidar_data, bbox, proj_points in lidar_data_for_objects:
        distance = bay_calculate_distance_to_object(obj_lidar_data, proj_points, bbox)
        if distance is not None:
            lidar_distances.append(distance)
        else:
            lidar_distances.append(0)
    lidar_distances = np.array(lidar_distances)

    # A prior distribution for the detection accuracy
    detection_accuracy_prior = tfd.Normal(loc=0.9, scale=0.1)
    detection_accuracies = detection_accuracy_prior.sample(sample_shape=len(detected_objects))

    # Bayesian model to calc log prob of the detection accuracy
    def model(detection_accuracies):
        log_prob_accumulator = 0.0
        for i, detection_accuracy in enumerate(detection_accuracies):
            expected_confidence = detection_accuracy * tf.where((lidar_distances[i] > 2) & (lidar_distances[i] < 30),
                                                                1.0, 0.5)
            observation_distribution = tfd.Bernoulli(probs=expected_confidence)
            log_prob_accumulator += observation_distribution.log_prob(detection_confidences[i])
        return log_prob_accumulator

    # MCMC sampling using Hamiltonian Monte Carlo and step size adaptation
    num_samples = 500
    num_burnin_steps = 50
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=model,
            num_leapfrog_steps=3,
            step_size=0.01),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    # Sample from the posterior distribution
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=[detection_accuracies],
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    # Calculate the mean of the posterior distribution FOR each detection
    detection_accuracy_posterior_samples = states[0]
    detection_accuracy_mean = np.mean(detection_accuracy_posterior_samples, axis=0)

    return detection_accuracy_mean

# def calculate_lidar_distances_for_objects(lidar_data, detected_objects, calib_params):
#     lidar_data_for_objects = extract_lidar_data_for_objects(lidar_data, detected_objects)
#     lidar_distances = [lidar_data['distance'] for lidar_data in lidar_data_for_objects]
#
#     return lidar_distances


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
        camera_image_with_fusion = visualize_fusion(camera_image, detected_objects, lidar_points, lidar_data_for_objects)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_processed_data(camera_image_with_fusion, lidar_points, output_dir, base_filename)


if __name__ == "__main__":
    main()
