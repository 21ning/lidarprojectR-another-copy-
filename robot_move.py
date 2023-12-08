import numpy as np
import random
import math

# transform the relative position to the pixel position
def Relative_Pixel(current_position, map_info):
    relative_x =  round((current_position[0] - map_info.origin[0]) / map_info.resolution )
    relative_y = round((current_position[1] - map_info.origin[1]) / map_info.resolution )
    position = np.array([relative_x, relative_y])
    return position

def Relative_Pixel2(current_position, resolution,origin ):
    relative_x =  round((current_position[0] - origin[0]) / resolution )
    relative_y = round((current_position[1] - origin[1]) / resolution )
    position = np.array([relative_x, relative_y])
    return position


# transform the pixel position to the relative position
def Pixel_Relative(current_position, map_info):
    relative_x =  current_position[0]*map_info.resolution + map_info.origin[0] 
    relative_y = current_position[1]*map_info.resolution + map_info.origin[1] 
    position = np.array([relative_x, relative_y])
    return position

#generate the random position for a pixel position
def Range_Relative(current_position, map_info):
    minx = max(current_position[0] - 0.5, 0)
    maxx = min(current_position[0] + 0.5, map_info.map_size[0])
    miny = max(current_position[1] - 0.5, 0)
    maxy = min(current_position[1] + 0.5, map_info.map_size[0])
    min_relative_x =  minx*map_info.resolution + map_info.origin[0] + 0.01
    max_relative_x =  maxx*map_info.resolution + map_info.origin[0] - 0.01
    min_relative_y = miny*map_info.resolution + map_info.origin[1] + 0.01
    max_relative_y = maxy*map_info.resolution + map_info.origin[1] - 0.01

    random_xvalue = np.random.uniform(min_relative_x, max_relative_x)
    random_yvalue = np.random.uniform(min_relative_y, max_relative_y)

    randon_point = np.array([random_xvalue, random_yvalue])
    return min_relative_x, max_relative_x, min_relative_y, max_relative_y,randon_point

# calcaulate the distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


# generate the robot position around the object position
def Cal_position(current_position, map_info):
    object_coverage = map_info.image_data
    min_distance = 0.96
    max_distance = 1.5
    while True:
        angle = random.uniform(0, 2 * 3.141592653589793)  
        distance = random.uniform(min_distance, max_distance)
        x = current_position[0] + distance * math.cos(angle)
        y = current_position[1] + distance * math.sin(angle)
        new_point = (x, y)
        pixel = Relative_Pixel(new_point, map_info)
        y = map_info.map_size[0] - pixel[0] - 1
        if np.all(pixel >= 0) and np.all(pixel < map_info.map_size):
            #if np.any(object_coverage[pixel[0],pixel[1]] != 0):
            if np.any(object_coverage[y,pixel[0]] != 0) and Check_position(new_point,map_info) == True:
               return new_point

# generate the robot position around the object position
def Check_position(current_position, map_info):
    a = Relative_Pixel2(current_position, map_info.resolution,map_info.origin )
    if 0<=a[0] < map_info.map_size[1] and 0<=a[1]<map_info.map_size[0]:
        return True



