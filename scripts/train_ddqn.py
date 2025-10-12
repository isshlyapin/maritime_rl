from maritime_rl.envs.maritime_env import MaritimeEnv
from maritime_rl.rl.agent import DDQNAgent
from maritime_rl.training.trainer import train
import torch

if __name__ == "__main__":
    env = MaritimeEnv(n_vessels=6, n_obstacles=3)
    obs_dim = env.obs_dim
    act_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(obs_dim, act_dim, device=device)
    train(env, agent, episodes=500)
