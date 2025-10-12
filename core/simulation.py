import math
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Vessel:
    x: float
    y: float
    v: float
    heading: float  # radians
    length: float = 50.0
    width: float = 10.0
    max_speed: float = 15.0
    max_turn_rate: float = math.radians(5)  # rad/s
    color: str = "blue"

    def step(self, throttle: float, rudder: float, dt: float):
        """Простейшая кинематика: throttle ∈ [-1,1], rudder ∈ [-1,1]."""
        dv = throttle * 0.2 * dt * self.max_speed
        dh = rudder * self.max_turn_rate * dt
        self.v = np.clip(self.v + dv, 0.0, self.max_speed)
        self.heading = (self.heading + dh) % (2 * math.pi)
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt

    def state_vector(self):
        return np.array([self.x, self.y, self.v, self.heading], dtype=np.float32)


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 50.0
    color: str = "gray"


@dataclass
class World:
    vessels: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    bounds: float = 2000.0  # world half-size, meters

    def step(self, actions: dict, dt: float):
        """actions: {vessel_idx: (throttle, rudder)}"""
        for i, vessel in enumerate(self.vessels):
            throttle, rudder = actions.get(i, (0.0, 0.0))
            vessel.step(throttle, rudder, dt)

    def check_collisions(self, ego_idx: int):
        ego = self.vessels[ego_idx]
        for i, v in enumerate(self.vessels):
            if i == ego_idx: continue
            if np.hypot(v.x - ego.x, v.y - ego.y) < (v.length + ego.length) * 0.5:
                return True
        for o in self.obstacles:
            if np.hypot(o.x - ego.x, o.y - ego.y) < o.radius:
                return True
        return False
