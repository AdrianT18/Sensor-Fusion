import os
import cv2
import numpy as np

"""
This script is used to preprocess the data from the KITTI dataset.
The goal of this script is to read the data from the dataset and then synchronize the data into pairs (LiDAR, Camera).
Next step is to remove the points that are not in the camera's field of view. Then we apply the calibration to the pairs.
Finally, we save the processed data to the output directory.
"""


# Loading camera data
def load_camera_image(file_path):
    image = cv2.imread(file_path)
    return image


# Loading lidar data
def load_lidar_data(file_path):
    lidar_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return lidar_data


# Here we read the calibration file and return the transformation matrix from the LiDAR to the camera and the camera matrix.
# This will be used to project the LiDAR points into the camera image. As both sensors will be calibrated using this file.
def read_kitti_calibration(calib_file_path):
    calib_params = {}

    # Here we read the calib file and then store the values in a dictionary.
    with open(calib_file_path, 'r') as file:
        for line in file:
            if ':' in line and line.split(':')[1].strip():
                try:
                    key, value = line.split(': ')
                    calib_params[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    continue

    # Here we get the transformation matrix from the LiDAR to the camera and the camera matrix.
    # P2 and Tr are the camera matrix taken from the left camera and the transformation matrix from the LiDAR to the camera.
    P_cam = calib_params.get('P2', np.eye(3, 4))
    T_velo_cam = np.vstack((calib_params.get('Tr', np.eye(4)[:3, :]).reshape(3, 4), [0, 0, 0, 1]))

    return T_velo_cam, P_cam


# This function aligns LiDAR data with the camera's perspective.
# 3 main steps:
# 1. Transforming LiDAR points to camera coordinates which ensures that a 4xN format is made after transformation.
# 2. Then we project these points onto the camera's image plane, converting 3D points to a 2D visual representation.
# 3. Filtering out points outside the camera's field of view to enhance system efficiency and accuracy.
# The result is a combination of 2D and 3D points, providing a comprehensive view from the camera's perspective.

def apply_calibration(lidar_points, Tr_velo_cam, P_cam):
    # Reshape the camera matrix for compatibility with LiDAR data (3X4)
    P_cam = P_cam.reshape(3, 4)

    # Create a homogeneous representation of LiDAR points
    num_points = lidar_points.shape[0]
    lidar_homogeneous = np.hstack((lidar_points[:, :3], np.ones((num_points, 1))))

    # Transform LiDAR points to camera coordinates
    lidar_cam = np.dot(Tr_velo_cam, lidar_homogeneous.T)
    if lidar_cam.shape[0] != 4:
        raise ValueError("Transformed LiDAR points (lidar_cam) should be a 4xN matrix.")

    # Project LiDAR points onto the image plane
    lidar_img = np.dot(P_cam, lidar_cam)

    # Filter points not in the camera's field of view
    in_front_of_camera = lidar_cam[2, :] > 0
    lidar_img = lidar_img[:, in_front_of_camera]
    lidar_cam = lidar_cam[:, in_front_of_camera]

    # Convert to 2D image coordinates
    lidar_img_2d = lidar_img[:2, :] / lidar_img[2, :]

    # Combine 2D image coordinates with LiDAR points
    combined_data = np.zeros((lidar_img_2d.shape[1], 4))
    combined_data[:, :2] = lidar_img_2d.T
    combined_data[:, 2:4] = lidar_cam[:2, :].T

    return combined_data, in_front_of_camera


# This function synchronizes data between camera and LiDAR directories based on timestamps. E.G, 000000.png and 000000.bin
# 3 main steps:
# 1. Reads all camera image and LiDAR data files, ensuring they are in sorted order.
# 2. Loads timestamps from a specified file.
# 3. Creates pairs of corresponding camera and LiDAR files aligned with each timestamp.
# The result is a list of tuples, each containing synchronized paths -> (camera_path, lidar_path, timestamp).

def synchronize_data(camera_dir, lidar_dir, timestamp_file):
    # A list to hold synchronized pairs
    synchronized_pairs = []

    # List and sort camera and LiDAR files
    camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.png')])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])

    # Read timestamps from the file - crucial for synchronization
    with open(timestamp_file, 'r') as f:
        timestamps = f.readlines()

    # Zip together ->  (camera_path, lidar_path, timestamp).
    for camera_file, lidar_file, timestamp in zip(camera_files, lidar_files, timestamps):
        camera_path = os.path.join(camera_dir, camera_file)
        lidar_path = os.path.join(lidar_dir, lidar_file)
        synchronized_pairs.append((camera_path, lidar_path, timestamp.strip()))

    return synchronized_pairs


# This function removes LiDAR points based on their projection onto the camera image and intensity.
# 3 main steps:
# 1. Apply calibration to project LiDAR points onto the camera image plane using the provided matrices.
# 2. Filter out LiDAR points that do not project intensely onto the camera image.
# 3. Create a mask to select points within the image boundaries and above the minimum intensity threshold.
# The result is a filtered set of LiDAR points that are visible in the camera image and meet intensity criteria.

def remove_points(lidar_points, camera_image, Tr_velo_cam2, P_rect2_cam2, min_intensity):
    # Project LiDAR points onto the camera image plane and filter valid projections
    projected_points, valid_projection_mask = apply_calibration(lidar_points, Tr_velo_cam2, P_rect2_cam2)
    lidar_points = lidar_points[valid_projection_mask]

    # Create a mask for points within image boundaries and above the intensity threshold
    # I choose to use the minimum intensity threshold of 0 to include all points. But can be changed in order
    # to filter out more points.
    mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < camera_image.shape[1]) & \
           (projected_points[:, 1] >= 0) & (projected_points[:, 1] < camera_image.shape[0]) & \
           (lidar_points[:, 3] >= min_intensity)

    return lidar_points[mask]


# This function visualizes LiDAR point projection on a camera image. It draws green circles at each LiDAR point's projected position.
# It extracts the x and y coordinates from the projected points and draws a circle at that position. (Green)
def visualize_lidar_camera_projection(camera_image, lidar_points, projected_points):
    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(camera_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)


# This function saves the processed data to the output directory.
def save_processed_data(camera_image_with_detections, lidar_points, output_dir, filename):
    print(f"Saving data to {output_dir} with filename {filename}")
    cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), camera_image_with_detections)
    lidar_points.tofile(os.path.join(output_dir, f"{filename}.bin"))


if __name__ == "__main__":
    image_dir = '../data/image_2_1'
    lidar_dir = '../data/velodyne_1'
    timestamp_file = '../data/timestamp/times.txt'
    calib_file_path = '../data/calib/calib.txt'
    output_dir = '../data/processed'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read calibration file
    T_velo_cam, P_cam = read_kitti_calibration(calib_file_path)

    # Synchronize data based on timestamps
    data_pairs = synchronize_data(image_dir, lidar_dir, timestamp_file)

    # A for loop to iterate through the data pairs and apply the calibration to each pair. Then visualize the results
    # and save the processed data to the output directory.
    for image_path, lidar_path, timestamp in data_pairs:
        camera_image = load_camera_image(image_path)
        lidar_points = load_lidar_data(lidar_path)
        projected_points, valid_projection_mask = apply_calibration(lidar_points, T_velo_cam, P_cam)
        visualize_lidar_camera_projection(camera_image, lidar_points[valid_projection_mask], projected_points)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_processed_data(camera_image, lidar_points, output_dir, filename)
