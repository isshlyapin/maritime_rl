import numpy as np
from tqdm import trange

def train(env, agent, episodes=1000, max_steps=500, eps_decay=0.995):
    eps = 1.0
    for ep in trange(episodes):
        s,_ = env.reset()
        total_r = 0
        for t in range(max_steps):
            a = agent.act(s, eps)
            ns,r,done,_,_ = env.step(a)
            agent.buffer.add(s,a,r,ns,float(done))
            agent.train_step()
            s = ns
            total_r += r
            if done: break
        eps = max(0.05, eps*eps_decay)
        if (ep+1)%10==0:
            print(f"Ep {ep+1} total reward={total_r:.3f}")
