import numpy as np
import open3d as o3d

"""
Check the distance of raw LIDAR points from first frame. Extra test.
"""


bin_path = '../data/velodyne_1/000000.bin'
point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

# Set the bounds for the points
x_min, x_max = 4.5, 15  # in front of the car
y_min, y_max = -5, 4  # driver's side of the car
z_min, z_max = -5, 4   # above the car

# Filter the points within the bounds
car_points = point_cloud[(point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
                         (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
                         (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)]

# Calculate the distances from the origin to the car points
distances = np.linalg.norm(car_points[:, :3], axis=1)

# Compute the average distance to the car
average_distance = np.mean(distances)


print(f"The average distance to the car is: {average_distance:.2f} meters")

# Convert the filtered points into an Open3D point cloud
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(car_points[:, :3])


o3d.visualization.draw_geometries([filtered_pcd])