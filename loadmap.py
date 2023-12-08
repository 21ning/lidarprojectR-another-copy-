import numpy as np
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os
import random
folder_path = "./maps" 
class MapData:
    def __init__(self, image_data, resolution, origin, negate, occupied_thresh, free_thresh , coverage):
        self.image_data = image_data
        self.resolution = resolution
        self.origin = origin
        self.negate = negate
        self.occupied_thresh = occupied_thresh
        self.free_thresh = free_thresh
        self.map_size = image_data.shape[:2]
        self.coverage = coverage
def load_map(map_file):
    with open(map_file, 'r') as f:
        map_data = yaml.safe_load(f)
    image_file = os.path.join(folder_path, map_data['image']) 
    resolution = map_data['resolution']
    origin = map_data['origin']
    negate = map_data['negate']
    occupied_thresh = map_data['occupied_thresh']
    free_thresh = map_data['free_thresh']
    image_data = plt.imread(image_file)
    map_size = image_data.shape[:2]
    coverage = image_data.copy()
    obstacle_pixels = np.argwhere(coverage == 0)
    Extra_obstacle = int(0.80/resolution)
    #Extra_obstacle = int(0.16/resolution)
    # add extra pixels around obstacles to get a better A star path
    for x, y in obstacle_pixels:
        for dx in range(-Extra_obstacle, Extra_obstacle+1):
            for dy in range(-Extra_obstacle, Extra_obstacle+1):
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < map_size[1] and 0 <= new_y < map_size[0]:
                    coverage[new_x, new_y] = 0
    return MapData(image_data, resolution, origin, negate, occupied_thresh, free_thresh, coverage)
