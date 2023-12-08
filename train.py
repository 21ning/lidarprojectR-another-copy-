import torch
import move
import matplotlib
matplotlib.use('Agg')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env,  SubprocVecEnv,DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime
from typing import Callable
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import cProfile
import tensorboardX as tbx
import time

def create_custom_env():
    return move.CustomEnv(reset_mode=1)
def create_test_env():
    return move.CustomEnv(reset_mode=2)
# visualize the environment
def visualize(agent, env, num_episodes=1):
    obs = env.reset()
    for _ in range(num_episodes): 
        done = False
        a =0
        while not done:
            a+=1
            action, _ = agent.predict(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            #env.render()
            #print(action*2)
            #if a%4==0:
            #   env.render()


def cal_finish(agent, env, num_episodes=100):
    obs = env.reset()
    a=0
    for _ in range(num_episodes): 
        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            #print(reward)
            if reward >1:
                a+=1
                #   print(a)
    #print(a/num_episodes)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_path = 'ppo_model.zip'
env_path = 'env'

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # acutuall callback frequency is callback_frequency*24, because all envs are sampled before calling
        self.save_frequency = 100000
        self.step_counter = 0
    def _on_step(self):
        self.step_counter +=1
        if self.step_counter % self.save_frequency == 0:
           current_time = datetime.datetime.now()
           formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
           model_dir = 'model'
           model_filename = f"ppo_model_{formatted_time}"
           env_filename = f"vec_normalize_{formatted_time}"
           env_path = os.path.join(model_dir, env_filename)
           model_path = os.path.join(model_dir, model_filename)
           ppo_agent.save(model_path)
           vec_env2.save(env_path)
        return True  

if __name__ == '__main__':
   custom_callback = CustomCallback()
   function_name = sys.argv[1]
   vec_env = make_vec_env(create_custom_env, n_envs=24, vec_env_cls=SubprocVecEnv)
   if function_name == "Render" :
      vec_env = make_vec_env(create_custom_env, n_envs=1, vec_env_cls=DummyVecEnv)
   trained_model = None
   #vec_env2 =  None
   vec_env3 = make_vec_env(create_custom_env, n_envs=1, vec_env_cls=DummyVecEnv)
   current_time = datetime.datetime.now()
   testvec_env = make_vec_env(create_test_env, n_envs=1, vec_env_cls=DummyVecEnv)
   formatted_time = current_time.strftime('%d_%H-%M')
   try:
       vec_env2 = VecNormalize.load(env_path, venv=vec_env)
       test_env = VecNormalize.load(env_path, venv=testvec_env)
       vec_env2.training = True 
       #每次2048*env 后打印一次
       ppo_agent= PPO.load(model_path, env=vec_env2,  verbose=1 , device="cuda", batch_size=2048)
       print("Model loaded successfully.")
   except FileNotFoundError:
       vec_env2 = VecNormalize(vec_env)
       test_env = VecNormalize(testvec_env)
       vec_env2.training = True 
       ppo_agent = PPO("MlpPolicy", vec_env2, device="cuda", batch_size=2048, tensorboard_log=f"./log/", verbose=1, gamma = 0.98, n_steps = 1024, seed =1)
       print(f"Model file '{model_path}' not found. Starting training from scratch.")
   if len(sys.argv) < 2:
        print("Usage: python my_script.py <function_name>")
        sys.exit(1)
   if function_name == "Train":
        a = 24*2048*200
        b = 1000000/24
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        ppo_agent.learn(total_timesteps=a, callback=custom_callback, reset_num_timesteps = False,  eval_env = test_env , eval_freq=10000,  eval_log_path = f"./log/",tb_log_name=formatted_time,n_eval_episodes=20)
   elif function_name == "Test":
        vec_env2.training = False 
        n_eval_episodes = 200# evaluate on 200 episodes
        #vec_env2 = VecNormalize.load(env_path, venv=vec_env)
        mean_reward, std_reward = evaluate_policy(ppo_agent, vec_env2, n_eval_episodes=n_eval_episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
   elif function_name == "Render":
        vec_env2.training = False 
        test_env.training = False
        #cal_finish(ppo_agent, vec_env2, num_episodes=100)
        #visualize(ppo_agent, test_env, num_episodes=100)
        #print("______________________________________________")
        visualize(ppo_agent, vec_env2, num_episodes=300)
   else:
    print("Invalid function name")

