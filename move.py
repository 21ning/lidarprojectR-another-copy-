import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import loadmap
import robot_move
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gym
import numpy as np
import lidar_generate3
import read_data
import os
import APF

map_info =read_data.choose_map("room1.yaml")
class CustomEnv(gym.Env):
    def __init__(self, reset_mode=1):
        super(CustomEnv, self).__init__()
        n = 32
        root = tk.Tk()
        root.title("Show")
# Find obstacle pixels and mark neighboring pixels
        fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.grouped_values = read_data.read_grouped_values_by_tag()
        if reset_mode == 2:
          #get the test map
          self.grouped_values = read_data.read_grouped_values_by_tag2()
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(n+4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Pixil position
        self.object_positions = [[]]
        # Relative position
        self.object_relative = [[]]
        self.object_velocities= [[]]
        self.robot_current = []
        self.times = 0
        self.maxspead = 0.6
        self.lidar_scan = [[]]
        self.encode_ring = [[]]
        self.map_info = None
        self.timecount=0
        self.difficulty_factor = 0.000001 
        #self.difficulty_factor = 4
        self.timecount2=0
        self.failcount=0
        self.failcount2=0
        self.failcount3=0
        self.human_max = 1.5
        self.human_mini= 0.96
        self.flex_range=0.3
        self.safe_policy_distance=0.8
        self.Rotation=0
        
    def reset(self):
      self.times= 0
      self.lidar_scan = [[]]
      self.robot_current = []
      self.object_positions = [[]]
      self.map_info = None
      self.raw_map_info = None
      self.timecount+=1
      # choose a set of motion data
      self.object_positions,self.object_velocities,self.object_relative, map_info = read_data.read_tag(self.grouped_values) 
      min_value =0
      #self.difficulty_factor += 0.000001
      self.map_info =  map_info
      self.raw_map_info =  map_info
      if self.difficulty_factor>1:
        self.map_info = read_data.modify_map(self.object_positions, map_info, self.difficulty_factor//1)
      # generate the robot start point
      while min_value < 0.64:
           self.robot_current= robot_move.Cal_position(self.object_relative[0], self.map_info)
           self.lidar_scan, self.encode_ring, min_value  = lidar_generate3.simulate_lidar_scan(self.robot_current, self.map_info, self.object_positions[0])
      relative_location=self.object_relative[0] - self.robot_current 
      velocity=self.object_velocities[0]
      velocity= velocity.flatten()
  
      #print(relative_location)
      observation = np.concatenate([
          self.encode_ring,
          velocity,
          relative_location,
      ])
      #print(observation)
      return observation
    def step(self, action):
        dt = 0.1
        if self.times < len(self.object_velocities)-1:
            # calculate the next position of the robot
            new_x = self.robot_current[0] + action[0] * dt * self.maxspead 
            new_y = self.robot_current[1] + action[1] * dt * self.maxspead 
            self.robot_current = np.array([new_x, new_y], dtype=np.float64)
            self.times += 1
            current_position = self.object_relative[self.times]
            relative_location =  current_position -  self.robot_current 
            dis= np.linalg.norm(relative_location) 
            self.lidar_scan, self.encode_ring, min_value = lidar_generate3.simulate_lidar_scan(self.robot_current, self.map_info, self.object_positions[self.times])  
            done = False 
            if 0.8 <= dis <= 1.8 and min_value>=0.64:
                relative_angle = np.arctan2(relative_location[1], relative_location[0])
                person_speed_angle = np.arctan2(self.object_velocities[self.times][1], self.object_velocities[self.times][0])
                robot_speed_angle_degrees = np.degrees(relative_angle)
                person_speed_angle_degrees = np.degrees(person_speed_angle)
                angle_difference2= abs(robot_speed_angle_degrees - person_speed_angle_degrees)
                scaled_difference = angle_difference2/180  
                # rence)
               # wall_distance=0
                wall_distance = 1.5 - min_value
                # -1，1
                abs_distance = 1- 10*abs(dis - 1.08)
                if(min_value>=3.0):
                  reward = 2 - 0.2 * scaled_difference
                else:
                  reward =  4*abs_distance - 2 *wall_distance
                if dis >1.5:
                   reward -= 4*abs_distance
                if(min_value<=1.5):
                   reward -= 2 *wall_distance
                reward = 0.1 *reward
                #if(min_value<0.8):
                #if(min_value<=1.0):
                #  wall_distance = 1.0 - min_value
                #  reward = 2 - wall_distance 
                #reward =  2*abs_distance - wall_distance
                
                #reward = 2*abs_distance -wall_distance
                #abs_distance = APF.reward_humancalculator.reward_for_person(dis) 
                #wall_distance = APF.reward_wallcalculator.reward_for_wall(min_value)
            else :
                done = True
                reward = -100
                if min_value< 0.64:
                   reward = -100
                   self.lidar_scan, self.encode_ring, min_value2 = lidar_generate3.simulate_lidar_scan(self.robot_current, self.raw_map_info, self.object_positions[self.times])  
                   if min_value2< 0.64:
                      a = "wall collision"
                      self.failcount+=1
                   else:
                      a = "obstacles collision"
                      self.failcount3+=1
                else:
                   a = 'out of human range'
                   self.failcount2+=1
                print(a)
                print('out of human range______________________________________')
                print(self.failcount2/self.timecount)
                print('wall collision______________________________________')
                print(self.failcount/self.timecount)
                print('obstacles collision______________________________________')
                print(self.failcount3/self.timecount)
#
            if self.times == len(self.object_velocities) - 1:
                done = True
                #self.timecount2+=1
                #print('finish')
                reward = 100
                #print(self.timecount2/self.timecount)
        velocity=self.object_velocities[self.times]
        velocity= velocity.flatten()
        observation = np.concatenate([
          self.encode_ring,
          velocity,
          relative_location,
      ])
        #print(velocity)
        return observation, reward, done, {}
    def render(self, mode=None, **kwargs):
        self.ax.clear()
        object_color = 'green'
        origin = self.map_info.origin
        origin = np.array(origin, dtype=np.double)
        map_data = self.map_info.image_data
        self.ax.imshow(map_data, extent=(
              origin[0], origin[0] + map_data.shape[1] * self.map_info.resolution,
              origin[1], origin[1] + map_data.shape[0] * self.map_info.resolution
          ))
        self.ax.plot(self.lidar_scan[:, 0], self.lidar_scan[:, 1], 'r.', markersize=0.5 , label='Lidar Scan',zorder=1)
        object_radius=0.16
        if self.times < len(self.object_positions):
           object_patch = Circle(self.object_positions[self.times]*map_info.resolution + origin[1],  object_radius, color=object_color, label='object')
        robot_color = "blue"
        robot_patch = Circle(self.robot_current, 0.48, color=robot_color, label='robot',zorder=3)
        self.ax.add_patch(object_patch)
        self.ax.add_patch(robot_patch)
        self.ax.set_title('Robot and Object Motion with Lidar Data')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#
    def close(self):
        pass

