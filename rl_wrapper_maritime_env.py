import numpy as np

from typing import List

from maritime_env import Ship
from maritime_env import MaritimeEnvironment

class RLWrapperMaritimeEnv:
    def __init__(self, map_size, max_steps, k_nearest: int, num_ships: int, ships: List['Ship'] | None = None):
        self.env = MaritimeEnvironment(map_size, max_steps, num_ships, ships)
        self.k_nearest = k_nearest

    def get_ship(self, ship_idx: int) -> Ship | None:
        return self.env.get_ship(ship_idx)

    def get_state_ship(self, ship_idx: int) -> np.ndarray | None:
        ship = self.get_ship(ship_idx)
        if ship is not None:
            return self._get_state(ship_idx)
        return None

    def _get_state(self, ship_idx: int) -> np.ndarray | None:
        ship = self.get_ship(ship_idx)
        if ship is not None:
            state = [
                (ship.target_x - ship.x) / self.env.size,   # normalized
                (ship.target_y - ship.y) / self.env.size,   # normalized
                ship.speed / ship.max_speed,                # normalized
                ship.heading / 180.0                        # normalized
            ]
            # Add information about K nearest ships
            nearest_ships = self.env.get_nearest_ships(ship_idx, self.k_nearest)
            for other_ship_idx in nearest_ships:
                other_ship = self.env.get_ship(other_ship_idx)
                if other_ship is not None:
                    state.extend([
                        ship.distance_to(other_ship) / self.env.size,   # normalized
                        ship.relative_speed(other_ship) / ship.max_speed, # normalized
                        ship.relative_heading(other_ship) / 180.0        # normalized
                    ])
            return np.array(state, dtype=np.float32)
        return None



def _get_state(self, ship_idx):
        """Get state representation for a ship"""
        ship = self.ships[ship_idx]
        state = [
            (ship['target_x'] - ship['x']) / self.size, # normalized 
            (ship['target_y'] - ship['y']) / self.size, # normalized
            ship['speed'] / self.max_speed,             # normalized
            ship['heading'] / 180.0                     # normalized
        ]
        
        # Add information about K nearest ships
        nearest_ships = self._get_k_nearest_ships(ship_idx)
        
        for nearest_idx in nearest_ships:
            other_ship = self.ships[nearest_idx]
            
            # Relative distance (normalized)
            dx = other_ship['x'] - ship['x']
            dy = other_ship['y'] - ship['y']
            distance = math.sqrt(dx**2 + dy**2) / self.size
            
            # Relative speed
            rel_speed = abs(self._relative_velocity(other_ship, ship)) / self.max_speed
            
            # Relative heading (angle difference)
            heading_diff = self._heading_diff(other_ship, ship) / 180.0
            
            state.extend([distance, rel_speed, heading_diff])
        
        # Pad with zeros if fewer than K nearest ships
        while len(state) < self.state_dim:
            state.extend([0, 0, 0])
        
        state_array = np.array(state, dtype=np.float32)
        
        # if verbose:
        #     print(f"  [STATE] Raw state (first 7 dims): {state_array[:7]}")
        #     print(f"  [STATE] State range: [{state_array.min():.3f}, {state_array.max():.3f}]")
        #     print(f"  [STATE] State mean: {state_array.mean():.3f}, std: {state_array.std():.3f}")
        
        return state_array

    def _calculate_collision_risk(self, ship_idx):
        """Calculate collision risk for a ship"""
        ship = self.ships[ship_idx]
        max_risk = 0
        
        for i, other_ship in enumerate(self.ships):
            if i == ship_idx:
                continue
                
            # Calculate distance
            dx = other_ship['x'] - ship['x']
            dy = other_ship['y'] - ship['y']
            distance = math.sqrt(dx**2 + dy**2)
            
            # Calculate relative speed
            rel_speed = abs(self._relative_velocity(other_ship, ship))
            
            # Collision risk formula from paper
            if distance <= self.d_safe_min:
                risk = -1.0
            elif self.d_safe_min < distance < self.d_safe_max:
                risk = (distance - self.d_safe_max) / (self.d_safe_max - self.d_safe_min) * (rel_speed / self.max_speed)
            else:
                risk = 0.0
            
            max_risk = min(max_risk, risk)

        return max_risk
    
    def _calculate_reward(self, ship_idx, state, action, next_state, verbose=False):
        """Calculate reward for a ship's action"""
        ship = self.ships[ship_idx]
        
        # Collision avoidance reward
        r_ca = self._calculate_collision_risk(ship_idx) * 10
        
        # Navigation efficiency reward
        dx = ship['target_x'] - ship['x']
        dy = ship['target_y'] - ship['y']
        desired_heading = math.degrees(math.atan2(dy, dx))

        d1 = math.sqrt(state[0]**2 + state[1]**2)
        d2 = math.sqrt(next_state[0]**2 + next_state[1]**2)
        
        dd = d1 - d2

        speed_dev = abs(ship['speed'] - ship['desired_speed']) / self.max_speed
        heading_dev = abs(self._normalize_angle_180(ship['heading'] - desired_heading)) / 180.0
        r_ne = - (speed_dev + heading_dev) / 2 + dd*100
        
        # COLREGs compliance reward (simplified)
        r_ce = self._check_colregs_compliance(ship_idx)
        
        # Combined reward with weights from paper
        # alpha, beta, gamma = 0.4, 0.4, 0.2
        # alpha, beta, gamma = 0, 1, 0
        alpha, beta, gamma = 0.5, 0.5, 0

        total_reward = alpha * r_ca + beta * r_ne + gamma * r_ce
        
        # if verbose:
        #     dist_to_target = math.sqrt(dx**2 + dy**2)
        #     print(f"  [REWARD] r_ca={r_ca:.3f}, r_ne={r_ne:.3f}, r_ce={r_ce:.3f}, total={total_reward:.3f}")
        #     print(f"  [STATE] speed={ship['speed']:.2f}, heading={ship['heading']:.1f}°, dist_to_target={dist_to_target:.1f}m")
        #     print(f"  [DEVIATIONS] speed_dev={speed_dev:.3f}, heading_dev={heading_dev:.3f}")
        
        return total_reward
    
    #TODO Ошибка реализации правил!!!
    def _check_colregs_compliance(self, ship_idx):
        """Simplified COLREGs compliance check"""
        # This is a simplified implementation
        # In practice, this would be more complex based on relative bearings, etc.
        # ship = self.ships[ship_idx]
        
        # for i, other_ship in enumerate(self.ships):
        #     if i == ship_idx:
        #         continue
                
        #     # Calculate relative bearing
        #     dx = other_ship['x'] - ship['x']
        #     dy = other_ship['y'] - ship['y']
        #     relative_bearing = math.degrees(math.atan2(dy, dx)) % 360
            
        #     # Simplified rule: avoid crossing from right
        #     if 0 <= relative_bearing <= 90:
        #         return 1.0  # Good compliance
        #     elif 270 <= relative_bearing <= 360:
        #         return 0.5  # Moderate compliance
        #     else:
        #         return 0.0  # Poor compliance
        
        return 0.0  # Default good compliance if no other ships
    
    def step(self, ship_idx, action, verbose=False):
        """Execute action for a ship and return next state, reward, done"""
        dt = 1.0  # time step in seconds

        state = self._get_state(0)

        for idx, ship in enumerate(self.ships):
            if idx == ship_idx:
                speed_idx = action // len(self.heading_changes)
                heading_idx = action % len(self.heading_changes)

                delta_speed = self.speed_changes[speed_idx]
                delta_heading = self.heading_changes[heading_idx]
            else:
                delta_speed = 0.0
                delta_heading = 0.0

            self._apply_ship_motion(ship, delta_speed, delta_heading, dt)
        
        next_state = self._get_state(0)
 
        # Calculate reward
        reward = self._calculate_reward(ship_idx, state, action, next_state, verbose=verbose)
        
        # Check if episode is done
        self.time_step += 1
        done = self.time_step >= self.max_steps
        termination_reason = None
        
        # Check for collisions
        if self._calculate_collision_risk(ship_idx) < (-1 + 1e-6):
            reward = -5
            done = True
            termination_reason = "collision"
        
        # Check achieving a goal
        ship = self.ships[ship_idx]
        target_d = math.sqrt(
            (ship['x'] - ship['target_x'])**2 +
            (ship['y'] - ship['target_y'])**2
        )
        if target_d < 5:
            reward = 100
            done = True
            termination_reason = "goal_reached"
            print(f"  [FINISH] Initial position: ({ship['x']:.1f}, {ship['y']:.1f})")
            print(f"  [FINISH] Target position: ({ship['target_x']:.1f}, {ship['target_y']:.1f})")

        
        if done and termination_reason is None:
            termination_reason = "max_steps"

        next_state = self._get_state(ship_idx, verbose=verbose)
        
        # if verbose:
        #     print(f"  [STEP] Action: speed_delta={delta_speed}, heading_delta={delta_heading}, reward={reward:.3f}, done={done}, reason={termination_reason}")
        
        return next_state, reward, done, termination_reason
    