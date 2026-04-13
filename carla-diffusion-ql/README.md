# 🚗 Autonomous Vehicle Navigation in CARLA
### A Reinforcement Learning Approach using Diffusion Q-Learning

> Mini Project — Department of Mechatronics, Manipal Institute of Technology (MAHE)  
> **Reva Shalu** (220929058) · **Alakhchandra Ladha** (220929186)

---

## 📽️ Demo

<!-- ADD YOUR DEMO VIDEO HERE -->
<!-- Option 1: Upload the video to this repo and replace the path below -->
<!-- Option 2: Upload to YouTube and paste the link below -->

> 🎬 **[Watch Demo Video](assets/demo.mp4)** ← *Replace with your video link or embed*

```
To embed a YouTube video, replace this block with:
[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

To use a local video file, place it in the assets/ folder and link it above.
```

---

## 📌 Overview

This project applies **Diffusion Q-Learning (DQL)** within the [CARLA](https://carla.org/) autonomous driving simulator to train an agent capable of:

- 🛣️ Lane following
- 🚧 Obstacle avoidance
- 🗺️ Route navigation in urban environments

The agent perceives the environment through **LiDAR, ego-state, waypoint, and lane information**, and controls the vehicle's **steering, throttle, and braking** in real time.

---

## 🧠 Algorithm: Diffusion Q-Learning (DQL)

DQL combines **diffusion-based policy generation** with **Q-learning** to produce smoother, more stable continuous control actions — a key advantage over standard RL methods for driving tasks.

| Component | Description |
|-----------|-------------|
| **Actor** | Diffusion model that generates actions from a noisy latent space |
| **Critic** | Twin Q-networks (Q1, Q2) evaluating state-action quality |
| **EMA** | Exponential Moving Average for stable target updates |
| **Optimizer** | Adam with optional Cosine Annealing LR scheduler |
| **Loss** | Behaviour cloning loss + Q-learning loss (weighted by `η`) |

---

## 📊 Results

Comparative evaluation against DDPG, TD3, and PPO across 4 metrics:

| Metric | DQL | DDPG | TD3 | PPO |
|--------|-----|------|-----|-----|
| **Avg Reward** | ✅ Highest | Low | Medium | Medium |
| **Collision Count** | ✅ Lowest (~3) | ~7 | ~5 | ~9 |
| **Lane Deviation** | ✅ ~0.18 m | ~0.35 m | ~0.30 m | ~0.40 m |
| **Success Rate** | ✅ **94%** | ~80% | ~85% | ~80% |

DQL outperforms all baselines — faster convergence, fewer collisions, tighter lane discipline, and highest task completion rate.

---

## 🗂️ Repository Structure

```
carla-diffusion-ql/
│
├── agents/
│   ├── ql_diffusion.py       # Diffusion_QL class (actor + critic + training loop)
│   ├── diffusion.py          # Diffusion model (denoising policy)
│   ├── model.py              # MLP backbone
│   └── helpers.py            # EMA utility
│
├── utils/
│   └── logger.py             # TensorBoard / logging utilities
│
├── params_dql/               # Saved model checkpoints (actor_200.pth, critic_200.pth)
│
├── assets/
│   └── demo.mp4              # ← Place your demo video here
│
├── run_simulation.py         # Main inference script — runs a single episode
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- CARLA Simulator 0.9.x ([download here](https://carla.org/))
- Python 3.8+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/carla-diffusion-ql.git
cd carla-diffusion-ql
pip install -r requirements.txt
```

### Start CARLA Server

```bash
# From your CARLA installation directory
./CarlaUE4.sh -carla-server   # Linux
CarlaUE4.exe                  # Windows
```

### Run the Simulation

```bash
python run_simulation.py
```

The script will load the pretrained model from `params_dql/` (checkpoint ID 200) and run one full episode on `Town10HD`.

---

## 🔧 Configuration

Key parameters in `run_simulation.py`:

```python
carla_params = {
    'number_of_vehicles': 10,
    'town': 'Town10HD',
    'desired_speed': 6,        # m/s
    'lidar_max_range': 50.0,   # meters
    'max_time_episode': 5000,  # timesteps
    'view_mode': 'follow',     # 'follow' or 'top'
    'traffic': 'off',          # 'on' for live traffic lights
}
```

---

## 📐 Observation Space

The state vector fed to the agent is **307-dimensional**, composed of:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| `ego_state` | 9 | Speed, heading, position |
| `lane_info` | 2 | Lateral offset, heading error |
| `lidar` | 240 | Discretized LiDAR scan |
| `nearby_vehicles` | 20 | Relative positions of up to 5 vehicles |
| `waypoints` | 36 | 12 future waypoints (x, y, heading) |

**Action space:** `[steering, throttle, brake]` ∈ [-1, 1]

---

## 📚 References

1. Dosovitskiy et al. (2017) — *CARLA: An Open Urban Driving Simulator* — CoRL
2. Lillicrap et al. (2016) — *DDPG: Continuous control with deep RL* — ICLR
3. Fujimoto et al. (2018) — *TD3: Addressing Function Approximation Error in Actor-Critic Methods* — ICML
4. Schulman et al. (2017) — *Proximal Policy Optimization Algorithms* — arXiv
5. Janner et al. (2022) — *Diffuser: Diffusion Models for Sequential Decision Making* — NeurIPS
6. Chen et al. (2023) — *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — arXiv

---

## 📄 License

This project was developed as an academic mini-project at MIT Manipal. For reuse or citation, please credit the authors.
