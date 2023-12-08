import numpy as np
from queue import PriorityQueue
import robot_move
import random

def a_star_search(map_info, start_position, target_position):
    # A star Algorithm 
    def get_neighbors(position, map_info):
        # get the neighbors of the current position
        neighbors = []
        # add the neighbors in the four directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x = position[0] + dx
            new_y = position[1] + dy
            if is_valid_position((new_x, new_y), map_info):
                neighbors.append((new_x, new_y))
                
        # add the neighbors in the four diagonal directions
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_x = position[0] + dx
            new_y = position[1] + dy
            if is_valid_position((new_x, new_y), map_info):
                neighbors.append((new_x, new_y))

        return neighbors
   # judge whether the position is valid
    def is_valid_position(position, map_info):
        x, y = position
        map_size = map_info.map_size
        y = map_size[0] - y - 1
        if 0 <= x < map_size[1] and 0 <= y < map_size[0]:
            #print(map_info.coverage.size)
            if map_info.coverage[y, x]!= 0:
                return True
        return False
        #return 0 <= x < map_size[0] and 0 <= y < map_size[1] and map_info.coverage[y, x]!= 0

    # A star Algorithm
    open_set = PriorityQueue()
    open_set.put((0, start_position))
    came_from = {}
    g_score = {(start_position[0], start_position[1]): 0}
    f_score = {(start_position[0], start_position[1]): np.linalg.norm(target_position - start_position)}

    while not open_set.empty():
        _, current = open_set.get()
        if np.array_equal(current, target_position):
            path = [current]
            while tuple(current) in came_from:
                current = came_from[tuple(current)]
                path.append(current)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, map_info):
            tentative_g_score = g_score[tuple(current)] + np.linalg.norm(np.array(neighbor) - np.array(current))
            if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                came_from[tuple(neighbor)] = tuple(current)
                g_score[tuple(neighbor)] = tentative_g_score
                f_score[tuple(neighbor)] = tentative_g_score + np.linalg.norm(target_position - np.array(neighbor))
                open_set.put((f_score[tuple(neighbor)], tuple(neighbor)))
    return None  

# gene a random target position, we give the range of the target position
def random_target(start, min_dist, map_info):
    """
    Generate a random target position.
    """
    dir = np.random.uniform(-1, 1, size=2)
    dir /= np.linalg.norm(dir)  # 归一化
    dist = np.random.uniform(min_dist, min_dist +5)/map_info.resolution  
    target_pos = start + np.round(dir * dist).astype(int)
    if target_pos[0] < 0 or target_pos[0] >= map_info.map_size[0] or target_pos[1] < 0 or target_pos[1] >= map_info.map_size[1]:
       return  False
    else:
        return target_pos

# generate a A star path 
def generate_feasible_target2(start_position, map_info, min_distance):
    """
    Generate a feasible target position and compute the path using A* search.
    """
    # set the max attempts to generate a feasible target position
    max_attempts = 100  
    for _ in range(max_attempts):
        target_position = random_target(start_position, min_distance, map_info)
        if  target_position is False:
            continue
        else:
            result = a_star_search(map_info, start_position, target_position)
            if result is not None:
             path = result
             abspath = [robot_move.Pixel_Relative(p, map_info) for p in path]
             return path, abspath
            
    # Failed to generate a A star path 
    return False


# Add noise to the path and return a set of motion trajectories
# The idea is to generate a path through the A-star algorithm to generate a target point in realistic coordinates 
# between every two pixel points and then move to that point.
# Then from that pixel point then calculate the position of the target point for the next pixel point.
def add_noise_to_velocities(abspath, path, map_info):
    noisy_velocities = []
    noisy_positions = []
    noisy_velocities2 = []
    noisy_positions2 = []

    dt = 0.1  # Time step

    # The noise factor, which is used to scale the speed, since the x or y speed 
    # calculated for every two pixels will be between (0.4m/s,0.8m/s) 
    # (0.4m/s,0.8m/s) because it can be assumed that due to rounding, 
    # a movement of half a pixel or one pixel will be considered as a movement to the next pixel point, 
    # e.g. (1,1) (1,2), which can be considered as a movement of half a resolution as well as a movement of one resolution.
    noise_factor = 0.5
    # Starting point
    current_point = abspath[0]  
    current_point2 = abspath[0]
    i = 1  # Loop counter
    noisy_positions2.append(current_point)
    
    # Attempts to generate a corresponding motion trajectory for each point of the path
    while i < len(path):

        judge = 0  
        # Loop until a satisfying point is found
        while judge == 0:
            time = 0
            time2 = 0
            noise = np.random.normal(0.95, 0.03)
            current_point = current_point2
            noisy_velocities = noisy_velocities2
            noisy_positions = noisy_positions2
            probability_of_zero = 0.1
            if random.random() < probability_of_zero:
             noise = 0.0
            # Calculate relative position and velocity of the next point
            min_relative_x, max_relative_x, min_relative_y, max_relative_y, next_point = robot_move.Range_Relative(path[i], map_info)
            velocity = next_point - current_point
            normalization_factor = velocity / np.linalg.norm(velocity) 

            velocity = noise_factor * normalization_factor *noise
            current_point = current_point + velocity * dt
            noisy_velocities.append(velocity)
            noisy_positions.append(current_point)

            path2 = []

            if i - 1 >= 0:
                path2.append(path[i - 1])
            path2.append(path[i])
            if i + 1 < len(path):
                path2.append(path[i + 1])

            # If velocity is too small, it will keep move until arrive the next pixel point, add a constraint to prevent the robot move beyond the path
            while current_point[0] < min_relative_x and current_point[1] < min_relative_y and time2 < 10:
                current_point = current_point + velocity * dt
                noisy_positions.append(current_point)
                time2 += 1

            pixiv = robot_move.Relative_Pixel(current_point, map_info)
            current_point2 = current_point

            # If the current point is out of path and then returning to the path
            while not np.any(np.all(pixiv == np.array(path2), axis=1)) and time < 100:
                time += 1
                min_relative_x, max_relative_x, min_relative_y, max_relative_y, next_point2 = robot_move.Range_Relative(path[i], map_info)
                velocity2 = next_point2 - current_point
                normalization_factor = velocity2 / np.linalg.norm(velocity2)
                velocity2 = noise_factor * normalization_factor
                current_point = current_point + velocity2 * dt
                pixiv = robot_move.Relative_Pixel(current_point, map_info)
                if time == 100:
                    current_point = current_point2

            # Check if the point on the path, if yes, move to the next point. Otherwise recalculate the motion trajectory of this point
            for j in range(len(path2)):
                if np.all(path2[j] == pixiv):
                    i = i + int(np.where(np.all(path2 == pixiv, axis=1))[0])
                    if time != 0:
                        noisy_velocities.append(velocity2)
                        noisy_positions.append(current_point)
                    noisy_velocities2 = noisy_velocities
                    noisy_positions2 = noisy_positions
                    current_point2 = current_point
                    judge = 1
                    break

    
    abspath2 = [robot_move.Relative_Pixel(p, map_info) for p in noisy_positions2]
    return abspath2, noisy_velocities2, noisy_positions2

