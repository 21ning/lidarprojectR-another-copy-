import math

class PotentialFieldReward:
    def __init__(self, k_attr, k_rep, desired_distance, d0 ):
        self.k_attr = k_attr
        self.k_rep = k_rep
        self.desired_distance = desired_distance
        self.d0 = d0

    def reward_for_person(self, distance_to_person):
        # Attractive potential to person
        attr_potential = 5* self.k_attr * abs(distance_to_person - self.desired_distance)

        # Repulsive potential from person if too close
        rep_potential_person = 0

        total_potential = attr_potential + rep_potential_person
        
        # The reward is the negative potential as we want to minimize it
        reward = -total_potential

        return reward
    
    def reward_for_wall(self, distance_to_wall):
        # Repulsive potential from wall
        rep_potential_wall = 0
        if distance_to_wall < self.d0:
            rep_potential_wall =self.k_rep * (1/distance_to_wall - 1/self.d0)

        # The reward is the negative potential as we want to minimize it
        reward = -rep_potential_wall

        return reward

# You can now initialize and compute rewards as:
reward_humancalculator = PotentialFieldReward(k_attr=1.0, k_rep=0.5, desired_distance=0.7, d0=0.7)
reward_wallcalculator = PotentialFieldReward(k_attr=1.0, k_rep= 1, desired_distance=5.0, d0=0.8)

#reward_person = reward_humancalculator.reward_for_person(0.9)
#print(f"Reward for person interaction: {reward_person}")
#reward_person = reward_humancalculator.reward_for_person(0.6)
#print(f"Reward for person interaction: {reward_person}")
#reward_person = reward_humancalculator.reward_for_person(0.5)
#print(f"Reward for person interaction: {reward_person}")
#reward_person = reward_humancalculator.reward_for_person(0.45)
#print(f"Reward for person interaction: {reward_person}")
#
#reward_wall = reward_wallcalculator.reward_for_wall(0.3)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(0.35)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(0.4)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(0.5)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(0.6)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(0.7)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(1.0)
#print(f"Reward for wall interaction: {reward_wall}")
#reward_wall = reward_wallcalculator.reward_for_wall(1.1)
#print(f"Reward for wall interaction: {reward_wall}")