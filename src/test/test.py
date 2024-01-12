import pandas as pd
import numpy as np

"""
Loads and aligns sensor fusion with ground truth data.
Calculates the positional error between the two.
"""


# Load sensor fusion
def load_sensor_fusion_output(output_file):
    data = np.loadtxt(output_file, delimiter=' ')
    positions = data[:1000, 1:4]
    return positions


# Load ground truth data
def load_ground_truth(ground_truth_file_path):
    data = np.loadtxt(ground_truth_file_path, delimiter=' ')
    transformation_matrices = data[:1000].reshape((-1, 3, 4))
    return transformation_matrices


# Load timestamps
def load_timestamps(timestamp_file):
    timestamps = np.loadtxt(timestamp_file)[:1000]
    return timestamps


# Align sensor fusion data with the ground truth data based on timestamps
def align_data(fusion_timestamps, fusion_positions, truth_timestamps, truth_positions):
    aligned_data = []
    for idx, timestamp in enumerate(fusion_timestamps):
        closest_idx = np.abs(truth_timestamps - timestamp).argmin()
        aligned_data.append((fusion_positions[idx], truth_positions[closest_idx]))
    return aligned_data


# Calculates the Euclidean distanced between the two
def calculate_errors(ekf_positions, truth_matrices):
    errors = []
    for i in range(len(ekf_positions)):
        if i >= len(truth_matrices):
            break
        truth_pos = truth_matrices[i, :, -1]
        error = np.linalg.norm(ekf_positions[i] - truth_pos[:2])
        errors.append(error)
    return errors


if __name__ == "__main__":
    sensor_fusion_output_file = '../data/processed/ekf_estimated_positions.txt'
    ground_truth_file = 'ground_truth/00.txt'
    timestamps = load_timestamps('../data/timestamp/times.txt')

    # Load the data
    # TODO - Change to load all the data
    ekf_positions = load_sensor_fusion_output(sensor_fusion_output_file)[:1000]
    truth_matrices = load_ground_truth(ground_truth_file)[:1000]
    truth_matrices.shape

    errors = calculate_errors(ekf_positions, truth_matrices)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"Mean positional error: {mean_error}")
    print(f"Standard deviation of positional error: {std_error}")
