# -*- coding: utf-8 -*-
"""
run_simulation.py
-----------------
Runs a single autonomous driving episode in the CARLA simulator using a
pretrained Diffusion Q-Learning (Diffusion_QL) agent.

Usage:
    python run_simulation.py
"""

import gym
import time
import easycarla
import numpy as np
import torch
import os
from agents.ql_diffusion import Diffusion_QL


# ===================== Helper Functions =====================

def convert_obs_dict_to_vector(obs_dict):
    """Convert observation dictionary to a flattened 307-dim state vector.

    Components:
        ego_state       : 9   dims — speed, heading, position
        lane_info       : 2   dims — lateral offset, heading error
        lidar           : 240 dims — discretized LiDAR scan
        nearby_vehicles : 20  dims — relative positions of up to 5 vehicles
        waypoints       : 36  dims — 12 future waypoints (x, y, heading)
    """
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions
        obs_dict['lane_info'],        # 2 dimensions
        obs_dict['lidar'],            # 240 dimensions
        obs_dict['nearby_vehicles'],  # 20 dimensions
        obs_dict['waypoints']         # 36 dimensions
    ]).astype(np.float32)


# ===================== Environment Configuration =====================

carla_params = {
    'number_of_vehicles': 10,
    'number_of_walkers': 0,
    'dt': 0.06,                            # time interval between two frames (s)
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    'surrounding_vehicle_spawned_randomly': True,
    'port': 2000,
    'town': 'Town10HD',
    'max_time_episode': 5000,              # maximum timesteps per episode
    'max_waypoints': 12,
    'visualize_waypoints': True,
    'desired_speed': 6,                    # desired speed (m/s)
    'max_ego_spawn_times': 200,
    'view_mode': 'follow',                 # 'top' or 'follow'
    'traffic': 'off',                      # 'on' = live traffic lights, 'off' = always green
    'lidar_max_range': 50.0,               # max LiDAR perception range (m)
    'max_nearby_vehicles': 5,
}


# ===================== Initialize Environment =====================

env = gym.make('carla-v0', params=carla_params)


# ===================== Initialize Model =====================

STATE_DIM  = 307
ACTION_DIM = 3
MAX_ACTION = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = Diffusion_QL(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    device=device,
    discount=0.99,
    tau=0.005,
    eta=0.01,
    beta_schedule='vp',
    n_timesteps=5,
)


# ===================== Load Pretrained Model =====================

MODEL_ID  = 200          # checkpoint ID
SAVE_PATH = './params_dql'

model.load_model(SAVE_PATH, id=MODEL_ID)
print(f"Successfully loaded model checkpoint ID {MODEL_ID} from '{SAVE_PATH}'")


# ===================== Run One Episode =====================

obs = env.reset()
done = False
step = 0
episode_reward = 0.0

while not done:
    obs_vec = convert_obs_dict_to_vector(obs)
    action  = model.sample_action(obs_vec)

    try:
        next_obs, reward, cost, done, info = env.step(action)
    except Exception as e:
        print(f"[Error] CARLA step failed: {e}")
        obs = env.reset()
        continue

    obs = next_obs
    episode_reward += reward
    step += 1

    # Uncomment to slow down playback for better visualization:
    # time.sleep(0.05)

print(f"\nEpisode finished.")
print(f"  Total reward : {episode_reward:.2f}")
print(f"  Total steps  : {step}")
