import argparse
import numpy as np
import cv2
import os
import torch
from EKF.data_preprocessing import load_camera_image, load_lidar_data, read_kitti_calibration, apply_calibration, \
    remove_points, synchronize_data
from EKF.kalman_filter import ExtendedKalmanFilter
from tqdm import tqdm
from yolov7.detect import detect
from yolov7.utils.plots import plot_one_box
from yolov7.models.experimental import attempt_load

"""
This is the main script for EKF sensor fusion.
The goal of this script is to fuse the data from the LiDAR and the camera using the Extended Kalman Filter. 
In the mean time we also detect objects in the camera image using YOLOv7. 
Then we also calculate the distance of the detected objects using the LiDAR data. 
Finally, save the fused data to the output directory.
"""


# Save the processed data to the output directory.
def save_processed_data(camera_image, lidar_points, output_dir, filename):
    cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), camera_image)
    lidar_points.tofile(os.path.join(output_dir, f"{filename}.bin"))


# This is a wrapper function for the detect function in detect.py
def detect_wrapper(opt):
    import yolov7.detect
    yolov7.detect.detect(opt)


# Performs object detection on an image and saves the results.
def object_detection_function(camera_image_with_lidar_overlay, model, output_dir):
    # Temp file of detected objects
    temp_image_path = os.path.join(output_dir, 'temp_detection_image.png')
    cv2.imwrite(temp_image_path, camera_image_with_lidar_overlay)

    # Options for yolov7 detection
    opt = argparse.Namespace()
    opt.source = temp_image_path
    opt.weights = 'yolov7.pt'
    opt.img_size = 640
    opt.conf_thres = 0.25
    opt.iou_thres = 0.45
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.view_img = False
    opt.save_txt = False
    opt.nosave = False
    opt.classes = None
    opt.agnostic_nms = False
    opt.augment = False
    opt.project = output_dir
    opt.name = 'detections'
    opt.exist_ok = True
    opt.save_img = True
    opt.no_trace = True

    # Perform detection
    detected_objects = detect(opt)

    # Read the image with objects detected
    detections_image = cv2.imread(os.path.join(output_dir, 'detections', os.path.basename(temp_image_path)))

    return detections_image, detected_objects


# This function calculates the distance to an object using the LiDAR data.
# Finds the LiDAR points that fall within the bounding box of a detected object,
# then calculates the distances of these points, and returns the average distance.
def calculate_distance_to_object(lidar_data, projected_points, bbox):
    # Bounding box coordinates
    x1, y1, x2, y2 = bbox

    # We find the LiDAR points that fall within the bbox
    associated_points_indices = [i for i, p in enumerate(projected_points) if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]

    # Calculate the average distance of the associated LiDAR points
    if associated_points_indices:
        associated_lidar_points = [lidar_data[i] for i in associated_points_indices]
        distances = [np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in associated_lidar_points]
        return np.mean(distances)
    return None


# This function stores the estimated position to a text file. Used to compare to ground truth data
def store_estimated_position(timestamp, estimated_position, output_dir):
    with open(os.path.join(output_dir, 'ekf_estimated_positions.txt'), 'a') as f:
        f.write(f"{timestamp} {estimated_position[0]} {estimated_position[1]}\n")


# This function runs the sensor fusion process on a set of data. It synchronizes the camera with LiDAR data,
# applies EKF for state estimation, performs object detection and saves the results.
# Future me - remove max frames for full dataset - TODO
def run_sensor_fusion(image_dir, lidar_dir, calib_file_path, output_dir, timestamp_file, batch_size=10,
                      max_frames=1000):
    # Initialization and setup of EKF
    temp_image_path = os.path.join(output_dir, 'temp_detection_image.png')
    timestamps = np.loadtxt(timestamp_file)

    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)
    WHEELBASE = 2.8  # meters

    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    state_covariance = np.diag([1, 1, 0.1, 1])
    process_noise = np.eye(4) * 0.1
    measurement_noise = np.eye(2) * 1

    # Make an instance of EKF with initial parameters ^^^
    ekf = ExtendedKalmanFilter(
        initial_state,
        state_covariance,
        process_noise,
        measurement_noise,
        WHEELBASE
    )

    # Load & synchronize data pairs.
    # TODO - remove max frames for full dataset (Second line)
    data_pairs = synchronize_data(image_dir, lidar_dir, timestamp_file)
    data_pairs = data_pairs[:max_frames]
    total_pairs = len(data_pairs)

    prev_position = None
    prev_timestamp = None

    # Load YOLOv7 model
    object_detection_model = attempt_load('yolov7.pt', map_location='cuda')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop through data pairs
    ekf_estimations = []
    for idx, (image_path, lidar_path, timestamp) in enumerate(tqdm(data_pairs, desc='Processing frames', unit='frame')):
        # Load and preprocess data
        camera_image = load_camera_image(image_path)
        lidar_points = load_lidar_data(lidar_path)
        lidar_points = remove_points(lidar_points, camera_image, T_velo_cam, P_cam, min_intensity=0.1)

        projected_points, _ = apply_calibration(lidar_points, T_velo_cam, P_cam)
        camera_image_with_lidar_overlay = camera_image.copy()
        # TODO - add this back in if you want to see the projected points
        # visualize_lidar_camera_projection(camera_image_with_lidar_overlay, lidar_points, projected_points)

        # Calculate velocity and predict state using EKF
        timestamp = float(timestamp)
        if prev_position is not None and prev_timestamp is not None:
            dt = timestamp - prev_timestamp
            if dt > 0:
                displacement = np.linalg.norm(ekf.state[:2] - prev_position)
                velocity = displacement / dt
            else:
                velocity = 0
        else:
            velocity = 0
            dt = 0.1

        # Predict EKF state
        prev_position = ekf.state[:2].copy()
        prev_timestamp = timestamp
        ekf.predict([0.0, velocity], dt)

        # Update EKF with new LiDAR data points
        for point in lidar_points:
            ekf.update(point[:2])

        # Save EKF estimation
        estimated_position = ekf.state[:2]
        store_estimated_position(
            timestamp,
            estimated_position,
            output_dir
        )

        # Object detection and annotation with distance information
        camera_image_with_detections, detected_objects = object_detection_function(
            camera_image_with_lidar_overlay,
            model,
            output_dir
        )

        for detected_object in detected_objects:
            bbox = detected_object['bbox']
            label = detected_object['label']

            distance = calculate_distance_to_object(lidar_points, projected_points, bbox)

            if distance is not None:
                label_with_distance = f'{label} | Dist: {distance:.2f}m'
                plot_one_box(
                    bbox,
                    camera_image_with_detections,
                    label=label_with_distance,
                    color=(0, 255, 0),
                    line_thickness=1
                )

        # Save processed data
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_processed_data(camera_image_with_detections, lidar_points, output_dir, base_filename)

    print(f"All data saved to {output_dir}.")


if __name__ == "__main__":
    # Use CPU if GPU is not available!!
    model = attempt_load('yolov7.pt', map_location='cuda')
    calib_file_path = '../data/calib/calib.txt'
    image_dir = '../data/image_2_1'
    lidar_dir = '../data/velodyne_1'
    calib_dir = '../data/calib'
    output_dir = '../data/processed'
    timestamp_file = '../data/timestamp/times.txt'

    # TODO - remove max frames for full dataset
    run_sensor_fusion(image_dir, lidar_dir, calib_file_path, output_dir, timestamp_file, batch_size=10, max_frames=1000)
