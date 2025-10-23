import numpy as np

from typing import List

import rl_config as cfg

from maritime_env import Ship
from maritime_env import MaritimeEnvironment

class RLWrapperMaritimeEnv:
    def __init__(self, 
            map_size, 
            max_steps,
            k_nearest: int, 
            num_ships: int,
            ships: List['Ship'] | None = None,
            test_idx = 0):
        
        self.env = MaritimeEnvironment(map_size, num_ships, ships)

        self.test_idx = test_idx

        self.k_nearest = k_nearest
        self.max_steps = max_steps

        self.speed_changes = cfg.SHIP_SPEED_CHANGES
        self.heading_changes = cfg.SHIP_HEADING_CHANGES

        self.target_radius = cfg.TARGET_RADIUS
        self.min_safe_distance = cfg.SHIP_MIN_SAFE_DISTANCE
        self.max_safe_distance = cfg.SHIP_MAX_SAFE_DISTANCE

    def get_state_ship(self, ship_idx: int) -> np.ndarray:
        try:
            return self._get_state(ship_idx)
        except Exception as e:
            raise

    def _get_state(self, ship_idx: int) -> np.ndarray:
        try:
            ship = self.env.get_ship(ship_idx)
            state = [
                (ship.target_x - ship.x) / self.env.size,   # normalized
                (ship.target_y - ship.y) / self.env.size,   # normalized
                ship.speed / ship.max_speed,                # normalized
                ship.heading / 180.0                        # normalized
            ]

            nearest_ships = self.env.get_nearest_ships(ship_idx, self.k_nearest)
            for other_ship_idx in nearest_ships:
                other_ship = self.env.get_ship(other_ship_idx)
                state.extend([
                    ship.distance_to(other_ship) / self.env.size,     # normalized
                    ship.relative_speed(other_ship) / ship.max_speed, # normalized
                    ship.relative_heading(other_ship) / 180.0         # normalized
                ])
            return np.array(state, dtype=np.float32)
        except Exception as e:
            raise

    def is_done(self) -> tuple[bool, str]:
        try:
            ship = self.env.get_ship(self.test_idx)
            if ship.distance_to_target() < self.target_radius:
                return True, "Reached target"
            if ship.x < -self.env.size or ship.x > self.env.size:
                return True, "Out of bounds"
            if ship.y < -self.env.size or ship.y > self.env.size:
                return True, "Out of bounds"

            nearest_ships = self.env.get_nearest_ships(self.test_idx, 1)
            if len(nearest_ships) > 0:
                nearest_ship = self.env.get_ship(nearest_ships[0])
                if ship.distance_to(nearest_ship) < self.min_safe_distance:
                    return True, "Collision"

            return False, ""
        except Exception as e:
            raise

    def _goal_reward(self, state1, state2) -> float:
        """Считает отношение разницы расстояний в двух состояниях к максимально возможному перемещению 
        (максимальное перемещение это effective_speed для корабля) отношение нужно ограничить 
        в интервале [-1, 1]"""
        distance_diff = np.linalg.norm(state1[:2]) - np.linalg.norm(state2[:2])
        effective_speed = self.env.get_ship(self.test_idx).effective_speed

        if effective_speed == 0:
            raise ValueError("Invalid effective speed")

        return max(-1, min(distance_diff / effective_speed, 1))
    
    def _speed_efficiency_reward(self, state1, state2):
        ship = self.env.get_ship(self.test_idx)
        effective_speed = ship.effective_speed / ship.max_speed
        
        if effective_speed == 0:
            raise ValueError("Invalid effective speed")

        ds1 = abs(state1[2] - effective_speed)
        ds2 = abs(state2[2] - effective_speed)

        return (ds1 - ds2) / abs(max(cfg.SHIP_SPEED_CHANGES, key=abs))

    def _collision_reward(self, state1, state2):
        ship = self.env.get_ship(self.test_idx)
        nearest_ships = self.env.get_nearest_ships(self.test_idx, self.k_nearest)

        max_risk = 0
        for i, other_ship_idx in enumerate(nearest_ships):
            nearest_ship = self.env.get_ship(other_ship_idx)
            distance = ship.distance_to(nearest_ship)
            rel_speed = ship.relative_speed(nearest_ship)
            risk = 0
            if distance <= self.min_safe_distance:
                risk = -1.0
            elif self.min_safe_distance < distance < self.max_safe_distance:
                risk = (distance - self.max_safe_distance) / (self.max_safe_distance - self.min_safe_distance) * (rel_speed / ship.max_speed)
            else:
                risk = 0.0
            
            max_risk = min(max_risk, risk)    
        
        return max_risk

    def _colregs_compliance_reward(self, state1, state2):
        pass

    def _calculate_reward(self, state1, action: int, state2) -> float:
        a = 0.4
        b = 0.2
        c = 0.4
        d = 0.1

        gr  = self._goal_reward(state1, state2)
        
        ser = self._speed_efficiency_reward(state1, state2)
        
        cr  = self._collision_reward(state1, state2)
        
        # ccr = self._colregs_compliance_reward(state1, state2)

        return a * gr + b * ser + c * cr

    def step(self, action: int):
        # Apply action to the ship
        if action < 0 or action >= len(self.speed_changes) * len(self.heading_changes):
            raise ValueError("Invalid action")

        speed_idx   = action // len(self.heading_changes)
        heading_idx = action %  len(self.heading_changes)

        state1 = self.get_state_ship(self.test_idx)

        self.env.apply_ship_motion(
             self.test_idx, 
             self.speed_changes[speed_idx], 
             self.heading_changes[heading_idx]
        )
        
        self.env.step(1)

        state2 = self.get_state_ship(self.test_idx)

        reward = self._calculate_reward(state1, action, state2)

        done, termination_reason = self.is_done()

        return state2, reward, done, termination_reason
