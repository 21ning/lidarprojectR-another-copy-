import numpy as np
import pythonring
from numba import jit
import time

@jit(nopython=True)  # Apply JIT compilation
def compute_points(position, map_data, resolution ,origin, object_position):
    # use blender to measure the robot base's width is 0.45m
    max_range = 10.20
    map_size = map_data.shape[:2]
    height, width = map_data.shape
    object_image_y = height - object_position[1] - 1  
    object_image_x = int(object_position[0])
    object_coverage = map_data.copy()
    object_coverage2 = map_data.copy()
    # 设置包括人的地图，碰到人激光雷达也会反射
    object_coverage[object_image_y, object_image_x] = 0
    object_coverage[object_image_y, object_image_x] = 0
    lidar_start_angle = np.pi
    num_samples = 360
    angles = np.linspace(lidar_start_angle, lidar_start_angle - 2* np.pi, num_samples)
    lidar_points2 = np.full(num_samples, 10.2) 
    # 
    lidar_points = []
    min_value = max_range/resolution + 1
    for i, angle in enumerate(angles):
        x, y = position
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        for time in np.arange(0, max_range, resolution):
            x += angle_sin * resolution
            y += angle_cos * resolution
            pixel_x = int((x - origin[0]) / resolution)
            pixel_y = map_size[0] - int((y - origin[1]) / resolution) - 1
            if 0 <= pixel_x < map_size[1] and 0 <= pixel_y < map_size[0]:
                # 原代码
                lidar_points.append([x, y])
                # 障碍物距离
                if not object_coverage2[pixel_y, pixel_x]:
                    if time < min_value:
                        min_value = time
                if not object_coverage[pixel_y, pixel_x]:
                    lidar_points.append([x, y])
                    lidar_points2[i] = time
                    break
            else:
                break
    lidar_data = np.array(lidar_points)
    lidar_points2 = lidar_points2.reshape(1, -1)
    lidar_points2 = lidar_points2.astype(np.float32)
    return lidar_data, lidar_points2, min_value


@jit(nopython=True)  # Apply JIT compilation
def get_dis(position, map_data, resolution ,origin):
    # use blender to measure the robot base's width is 0.45m
    min_range = 0.48
    max_range = 10.20
    map_size = map_data.shape[:2]
    height, width = map_data.shape
    object_coverage2 = map_data.copy()
    lidar_start_angle = np.pi
    num_samples = 360
    angles = np.linspace(lidar_start_angle, lidar_start_angle - 2* np.pi, num_samples)
    min_value = max_range/resolution + 1
    for i, angle in enumerate(angles):
        x, y = position
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        for time in np.arange(0, max_range, resolution):
            x += angle_sin * resolution
            y += angle_cos * resolution
            pixel_x = int((x - origin[0]) / resolution)
            pixel_y = map_size[0] - int((y - origin[1]) / resolution) - 1
            if 0 <= pixel_x < map_size[1] and 0 <= pixel_y < map_size[0]:
                #障碍物距离
                if not object_coverage2[pixel_y, pixel_x]:
                    if time < min_value:
                        min_value = time
                        break
            else:
                break
    min_value = min_value * resolution
    return  min_value


def simulate_lidar_scan(position, map_info, object_position):
    map_data = map_info.image_data
    resolution = map_info.resolution
    origin = np.array(map_info.origin)
    lidar_data, lidar_points2, min_value = compute_points(position, map_data, resolution, origin, object_position)
    rings_compress = pythonring.ring_def["encoder2"](lidar_points2,0)
    return lidar_data, rings_compress.flatten()  , min_value

