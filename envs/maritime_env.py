import gymnasium as gym
from gymnasium import spaces
import numpy as np
from maritime_rl.core.simulation import World, Vessel, Obstacle
from maritime_rl.envs.rewards import compute_reward

class MaritimeEnv(gym.Env):
    """Совместимая с Gym среда."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_vessels=5, n_obstacles=2, k_nearest=4, dt=1.0):
        super().__init__()
        self.dt = dt
        self.world = World()
        self.k = k_nearest
        self.n_vessels = n_vessels
        self.n_obstacles = n_obstacles
        self._build_world()
        # actions: throttle ∈ [-1,1], rudder ∈ [-1,1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # observation: ego + K nearest [x,y,v,h, rel_dx,rel_dy,rel_dv,rel_dh]
        self.obs_dim = 4 + self.k * 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.step_count = 0
        self.max_steps = 500

    def _build_world(self):
        self.world.vessels = [
            Vessel(x=np.random.uniform(-500,500),
                   y=np.random.uniform(-500,500),
                   v=np.random.uniform(4,8),
                   heading=np.random.uniform(0,2*np.pi),
                   color="red" if i==0 else "blue")
            for i in range(self.n_vessels)
        ]
        self.world.obstacles = [
            Obstacle(x=np.random.uniform(-800,800),
                     y=np.random.uniform(-800,800),
                     radius=np.random.uniform(50,100))
            for _ in range(self.n_obstacles)
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_world()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # apply ego action
        actions = {0: (action[0], action[1])}
        self.world.step(actions, self.dt)
        # other vessels: simple straight motion
        for i in range(1, self.n_vessels):
            self.world.vessels[i].step(0, 0, self.dt)
        # compute reward
        done = self.world.check_collisions(0)
        reward = compute_reward(self.world, ego_idx=0)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        ego = self.world.vessels[0]
        ego_vec = ego.state_vector()
        others = []
        for i, v in enumerate(self.world.vessels[1:], start=1):
            dx, dy = v.x - ego.x, v.y - ego.y
            dist = np.hypot(dx, dy)
            others.append((dist, v))
        others.sort(key=lambda x: x[0])
        obs = np.zeros(self.obs_dim, np.float32)
        obs[:4] = ego_vec
        idx = 4
        for d,v in others[:self.k]:
            rel = np.array([v.x-ego.x, v.y-ego.y, v.v-ego.v, (v.heading-ego.heading)], np.float32)
            obs[idx:idx+4] = rel
            idx += 4
        return obs

    def render(self):
        print(f"step {self.step_count} ego=({self.world.vessels[0].x:.1f},{self.world.vessels[0].y:.1f})")

    def close(self):
        pass
