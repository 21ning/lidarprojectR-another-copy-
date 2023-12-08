import numpy as np
import os
import pickle
import random
import loadmap
import copy
import lidar_generate3
folder_path = "./maps" 
all_yaml_files = [f for f in os.listdir(folder_path) if f.endswith('.yaml')]
map_infos = {}
for map_file in all_yaml_files:
    full_path = os.path.join(folder_path, map_file)
    map_info = loadmap.load_map(full_path)
    map_infos[map_file] = map_info

def read_grouped_values_by_tag():
    parent_directory = 'move_folder'
    sub_directory2 = os.path.join(parent_directory, 'object_move')
    file_path = os.path.join(sub_directory2, 'noise_data.pkl')
    grouped_values = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                grouped_values.append([data['path'], data['velocities'], data['noisy_positions'],data['map_name']])
        except EOFError:
            pass
    return  grouped_values

def read_grouped_values_by_tag2():
    parent_directory = 'test_folder'
    sub_directory2 = os.path.join(parent_directory, 'object_move')
    file_path = os.path.join(sub_directory2, 'noise_data.pkl')
    grouped_values = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                grouped_values.append([data['path'], data['velocities'], data['noisy_positions'],data['map_name']])
        except EOFError:
            pass
    return  grouped_values

def choose_map(chosen_map_file):
    chosen_map_info = map_infos[chosen_map_file]
    return  chosen_map_info

# random select a set of data
def read_tag2(grouped_values):
    a = random.randint(0, len(grouped_values)-1)
    path,velocities,noisy_positions, map_name = grouped_values[a]
    map_info = choose_map(map_name)
    #map_info = choose_map(map_name+".yaml")
    return np.array(path), np.array(velocities), np.array(noisy_positions), map_info
def read_tag(grouped_values):
    a = random.randint(0, len(grouped_values)-1)
    path, velocities, noisy_positions, map_name = grouped_values[a]
    map_info = choose_map(map_name)  # 假设 choose_map 是一个已定义的函数
    # 对 path, velocities, noisy_positions 进行倒序并与原始数组拼接
    path_combined = np.concatenate((path, path[::-1]))
    velocities_combined = np.concatenate((velocities, velocities[::-1]))
    noisy_positions_combined = np.concatenate((noisy_positions, noisy_positions[::-1]))

    if random.random() < 0.5:  # 50% chance
       path_combined = path_combined[::-1]
       velocities_combined = velocities_combined[::-1]
       noisy_positions_combined = noisy_positions_combined[::-1]

    return path_combined, velocities_combined, noisy_positions_combined, map_info


def random_end_slice(arr,sample_size):
    """Selects a random slice from an array starting from a random point to the end."""
    start = np.random.randint(0, len(arr) - sample_size)
    return arr[start:start + sample_size]


def read_paths():
    parent_directory = 'move_folder'
    sub_directory2 = os.path.join(parent_directory, 'path')
    file_path = os.path.join(sub_directory2,'data.pkl')
    grouped_values = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                grouped_values.append([data['object_positions'], data['abspath'],data['map_name']])
        except EOFError:
            pass
    return grouped_values

def getmap_infos():
    return  map_infos

# auto ajdust the obstacles depend on the difficulty_factor
def modify_map(object_positions, map_info2, difficulty_factor = 0):
    map_info = copy.deepcopy(map_info2)
    object_positions = list(object_positions)
    map_data = map_info.image_data
    resolution = map_info.resolution
    origin = map_info.origin
    fix_dis = 0.96
    #difficulty_factor 
    #if difficulty_factor>4:
    #   difficulty_factor = 4 
    random_number = random.randint(0, difficulty_factor)
    random_number2 = random.randint(0, difficulty_factor)
    random_number3 = random.randint(0, difficulty_factor)
    # a = 0
    # add small obstacles
    # add bigger obstacles
    if random_number2 > 0:
        point_pairs = [(object_positions[i], object_positions[i+1]) for i in range(0, len(object_positions) - 1, 2)]
        valid_pairs = [pair for pair in point_pairs if all(lidar_generate3.get_dis(point, map_info.image_data, resolution ,origin) > fix_dis for point in pair)]
        selected_pairs = random.sample(valid_pairs, min(random_number2, len(valid_pairs)))
        for selected_pair in selected_pairs:
            for x, pixel_y in selected_pair:
                pixel_y = map_info.map_size[0] - pixel_y - 1
                if 0 <= x < map_data.shape[1] and 0 <= pixel_y < map_data.shape[0]:
                    map_info.image_data[pixel_y, x] = 0 
    
    # add bigger obstacles
    if random_number3 > 0:
        point_pairs = [(object_positions[i], object_positions[i+1]) for i in range(0, len(object_positions) - 1, 3)]
        valid_pairs = [pair for pair in point_pairs if all(lidar_generate3.get_dis(point, map_info.image_data, resolution ,origin) > fix_dis for point in pair)]
        selected_pairs = random.sample(valid_pairs, min(random_number3, len(valid_pairs)))    

        for selected_pair in selected_pairs:
            for x, pixel_y in selected_pair:
                pixel_y = map_info.map_size[0] - pixel_y - 1
                if 0 <= x < map_data.shape[1] and 0 <= pixel_y < map_data.shape[0]:
                    map_info.image_data[pixel_y, x] = 0 
    
    selected_points = []
    i =0
    while i < random_number:
      point = random.choice(object_positions)
      if lidar_generate3.get_dis(point, map_data, resolution ,origin) > fix_dis:
          pixel_y = map_info.map_size[0] - point[1] - 1
          x = point[0]
          map_info.image_data[pixel_y, x] = 0 
          i+=1
    return map_info