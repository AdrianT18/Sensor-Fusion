import os
from haversine import haversine, Unit


# Read oxts file
def read_oxts(file_path):
    data = []
    with open(file_path, 'r') as file:
        line = file.readline()
        tokens = line.split()
        frame_data = {
            'latitude': float(tokens[0]),
            'longitude': float(tokens[1]),
            'altitude': float(tokens[2]),
        }
    return frame_data


# Process oxts data
def process_oxts(folder_path):
    oxts_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            frame_number = int(file_name.split('.')[0])
            file_path = os.path.join(folder_path, file_name)
            oxts_data[frame_number] = read_oxts(file_path)
    return oxts_data


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    # Calculate distance using the haversine library
    # The result is returned in miles, which is then converted to meters
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.MILES) * 1609.34


def calculate_distance_between_frames(frame1_path, frame2_path):
    frame1_data = read_oxts(frame1_path)
    frame2_data = read_oxts(frame2_path)

    lat1 = frame1_data['latitude']
    lon1 = frame1_data['longitude']
    lat2 = frame2_data['latitude']
    lon2 = frame2_data['longitude']

    return calculate_haversine_distance(lat1, lon1, lat2, lon2)


# Example usage
frame0_path = './oxts/0000000147.txt'
frame5_path = './oxts/0000000170.txt'

distance = calculate_distance_between_frames(frame0_path, frame5_path)
print(f"Distance between Frame 0 and Frame 5: {distance} meters")