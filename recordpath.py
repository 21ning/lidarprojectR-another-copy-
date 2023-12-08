import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import Astar
import datetime
import os
import sys
import pickle
import read_data
import multiprocessing


folder_path = "./maps" 


#map_name = 'asl'
#map_name = 'room1'
#map_name = 'room2'

# Find obstacle pixels and mark neighboring pixels
fig, ax = plt.subplots(figsize=(8, 8))
#generate 100 A star path 
#Add small obstacles to the map

#保存障碍物的文件
#image_file2 = os.path.join(folder_path, "asl2.pgm") 
#image_data2 = plt.imread(image_file2)
def Recordpath(map_name):
    map_info = read_data.choose_map(map_name)
    coverage = map_info.coverage
    object_positions = [[]]
    i=0
    parent_directory = 'move_folder'
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    sub_directory = os.path.join(parent_directory, 'object_move')
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    sub_directory2 = os.path.join(parent_directory, 'path')
    if not os.path.exists(sub_directory2):
        os.makedirs(sub_directory2)
    save_path_data = os.path.join(sub_directory2, 'data.pkl')
    while i<100:
          Pixel_position = np.random.randint(low=(0, 0), high=map_info.map_size, size=2)
          while coverage[Pixel_position[0],Pixel_position[1]] != 255:
              Pixel_position = np.random.randint(low=(0, 0), high=map_info.map_size, size=2)       
          result = Astar.generate_feasible_target2(Pixel_position, map_info, 15)
          if result==False:
              continue
          else:
              i+=1
              object_positions,abspath= result              
              with open(save_path_data, 'ab') as f:
                  pickle.dump({'object_positions': object_positions, 'abspath': abspath,'map_name':map_name}, f)
                  #print(map_name)
              #print(f"A star path saved {i}")
              #render(object_positions, sub_directory2,map_name,map_info)
    print(map_name+"300 finish")
# save the path picture
def render(path, save_directory,map_name,map_info3):
    ax.clear()
    map_data = map_info3.image_data
    modified_map_data = map_data.copy()
    origin = map_info3.origin
    origin = np.array(origin, dtype=np.double)
    for pixel_x, pixel_y in path:
        pixel_y = map_info3.map_size[0] - pixel_y - 1
        if 0 <= pixel_x < map_data.shape[1] and 0 <= pixel_y < map_data.shape[0]:
            modified_map_data[pixel_y, pixel_x] = 1 
    ax.imshow(modified_map_data, extent=(
             origin[0] ,origin[0] + map_data.shape[1] * map_info3.resolution,
             origin[1] + map_data.shape[0] * map_info3.resolution , origin[1], 
         ))
    ax.set_title('Robot and Object Motion with Lidar Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("-%M-%S")
    save_path = os.path.join(save_directory, map_name + f"{formatted_datetime}.png")
    plt.savefig(save_path)
    
def render2(save_directory,map_name,map_info3):
    ax.clear()
    map_data = map_info3.image_data
    modified_map_data = map_data.copy()
    origin = map_info3.origin
    origin = np.array(origin, dtype=np.double)
    ax.set_title('Robot and Object Motion with Lidar Data')
    ax.imshow(modified_map_data, extent=(
             origin[0] ,origin[0] + map_data.shape[1] * map_info3.resolution,
             origin[1] + map_data.shape[0] * map_info3.resolution , origin[1], 
         ))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("-%M-%S")
    save_path = os.path.join(save_directory, map_name + f"{formatted_datetime}.png")
    plt.savefig(save_path)
    
#Add noise to the every path
def Recordnoise():
    object_positions = [[]]
    i = 0
    parent_directory = 'move_folder'
    grouped_values = read_data.read_paths()
    a = len(grouped_values) 
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    sub_directory = os.path.join(parent_directory, 'object_move')
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    save_path_data = os.path.join(sub_directory, 'noise_data.pkl')
    while i < a:
        #h = random.randint(0, 99)
        h=i
        object_positions, abspath, map_name  = grouped_values[h]
        map_info2 = read_data.choose_map(map_name)
        path, velocities, noisy_positions = Astar.add_noise_to_velocities(abspath, object_positions, map_info2)
        i += 1
        with open(save_path_data, 'ab') as f:
            data = {'path': path, 'velocities': velocities, 'noisy_positions': noisy_positions, 'map_name':map_name}
            pickle.dump(data, f)
    print(f" {i} Data saved to: {save_path_data}")

def worker(map_name2):
    Recordpath(map_name2)
def process_map(_):  # 参数下划线仅作为一个占位符
    map_name2 = 'room9.yaml'
    Recordpath(map_name2)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python my_script.py <function_name>")
        sys.exit(1)
    
    function_name = sys.argv[1]
    if function_name == "Recordpath":
        #map_infos = read_data.getmap_infos()
        #with multiprocessing.Pool(processes=24) as pool:
        #     pool.map(worker, map_infos)
        #Recordnoise()
        map_name2 = 'room9.yaml'
        Recordpath(map_name2)
       #N = 10  # 例如，你想并行运行10次
       #with multiprocessing.Pool(processes=24) as pool:
       # pool.map(process_map, range(N))

    elif function_name == "Recordnoise":
        Recordnoise()
    elif function_name == "Recordnoise2":
        map_name2 = 'room9.yaml'
        Recordpath(map_name2)
        Recordnoise()
    else:
        print("Invalid function name")


