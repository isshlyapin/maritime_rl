import numpy as np

def compute_reward(world, ego_idx=0, dsafe=50.0, dmax=500.0,
                   alpha=1.0, beta=0.3, gamma=0.3):
    ego = world.vessels[ego_idx]
    dmin = min(np.hypot(ego.x-v.x, ego.y-v.y)
               for i,v in enumerate(world.vessels) if i!=ego_idx)
    if dmin < dsafe:
        r_ca = -1.0
    elif dmin < dmax:
        r_ca = (dmin-dsafe)/(dmax-dsafe)
    else:
        r_ca = 0.0
    r_ne = -abs(ego.v - 8.0)/8.0
    r_cc = 1.0
    return alpha*r_ca + beta*r_ne + gamma*r_cc
