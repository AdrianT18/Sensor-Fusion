import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" 
Makes 3D plot of EKF estimated positions and ground truth positions.
Helps me verify how well the EKF is doing.
"""

# Loads EKF estimated positions
def load_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


# Loads ground truth data
def process_ground_truth(ground_truth_data):
    try:
        num_rows, num_cols = ground_truth_data.shape
        if num_cols != 12:
            raise ValueError(f"Expected 12 columns per line for ground truth data, found {num_cols}.")
        return ground_truth_data.reshape((-1, 3, 4))[:, :, -1]
    except Exception as e:
        print(f"Error processing ground truth data: {e}")
        return None


# Plots EKF estimated positions and ground truth positions
def plot_positions(ekf_positions, truth_positions):
    try:
        # New fig for 3D plot
        fig = plt.figure()

        # 3D subplot to fig
        ax = fig.add_subplot(111, projection='3d')

        # Plot EKF estimated positions and ground truth positions
        ax.scatter(ekf_positions[:, 0], ekf_positions[:, 1], ekf_positions[:, 2], c='r', marker='o', label='EKF Estimated')
        ax.scatter(truth_positions[:, 0], truth_positions[:, 1], truth_positions[:, 2], c='g', marker='^', label='Ground Truth')

        # TODO- Add title, axis labels, legend
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()

        plt.show()
    except Exception as e:
        print(f"Error plotting positions: {e}")


if __name__ == "__main__":
    ekf_positions_path = '../data/processed/ekf_estimated_positions.txt'
    ground_truth_file_path = 'ground_truth/00.txt'

    ekf_positions = load_data(ekf_positions_path)
    ground_truth_data = load_data(ground_truth_file_path)

    if ekf_positions is not None and ground_truth_data is not None:
        # TODO change this for full dataset
        truth_positions = process_ground_truth(ground_truth_data)[:1000]

        # TODO change this for full dataset
        ekf_positions = ekf_positions[:1000]

        if truth_positions is not None and ekf_positions.shape[0] == truth_positions.shape[0]:
            print(f"EKF positions shape: {ekf_positions.shape}")
            print(f"Ground truth positions shape: {truth_positions.shape}")
            plot_positions(ekf_positions, truth_positions)
        else:
            print(
                f"Data size mismatch: EKF positions count {ekf_positions.shape[0]}, Ground truth matrices count {truth_positions.shape[0]}"
            )
    else:
        print("Failed to process ground truth data.")