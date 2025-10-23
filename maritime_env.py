import math
import random

from typing import List

MAX_SPEED_RANDOM_SHIP = 10
EFFECTIVE_SPEED_RANDOM_SHIP = 7


def normalize_angle_180(angle_deg):
    """Приводит угол к диапазону [-180, 180]"""
    return (angle_deg + 180) % 360 - 180


class Ship:
    def __init__(self, x, y, target_x, target_y, speed, heading, effective_speed, max_speed):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.speed    = speed
        self.heading  = heading
        self.max_speed       = max_speed
        self.effective_speed = effective_speed

    def distance_to(self, other_ship):
        dx = other_ship.x - self.x
        dy = other_ship.y - self.y
        return math.sqrt(dx**2 + dy**2)

    def distance_to_target(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return math.sqrt(dx**2 + dy**2)

    def relative_speed(self, other_ship):
        dvx = other_ship.speed * math.cos(math.radians(other_ship.heading)) - self.speed * math.cos(math.radians(self.heading))
        dvy = other_ship.speed * math.sin(math.radians(other_ship.heading)) - self.speed * math.sin(math.radians(self.heading))
        return math.sqrt(dvx**2 + dvy**2)
    
    def relative_heading(self, other_ship):
        return normalize_angle_180(other_ship.heading - self.heading)
    
    def change_heading(self, delta_head):
        self.heading = normalize_angle_180(self.heading + delta_head)

    def change_speed(self, delta_speed):
        self.speed = min(max(self.speed + delta_speed, 0), self.max_speed)

    def go_straight(self, dt):
        self.x += self.speed * math.cos(math.radians(self.heading)) * dt
        self.y += self.speed * math.sin(math.radians(self.heading)) * dt

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "speed":   self.speed,
            "heading": self.heading,
            "max_speed":       self.max_speed,
            "effective_speed": self.effective_speed
        }


class MaritimeEnvironment:
    def __init__(self, map_size, num_ships: int, ships: List['Ship'] | None = None):
        # Параметры карты
        self.size = map_size
        
        # Параметры симуляции
        self.time_step = 0

        self.ships = ships
        self.num_ships = num_ships

        self.reset(self.ships)

        if not self.is_valid():
            raise ValueError("Invalid environment configuration")

    def is_valid(self):
        if self.ships is None or len(self.ships) != self.num_ships:
            return False
        
        return True

    def reset(self, ships: List['Ship'] | None = None):
        """Reset environment to initial state"""
        self.time_step = 0
        
        # Добавление кораблей в окружение
        if ships is None:
            self.ships = []
            for i in range(self.num_ships):
                self.ships.append(
                    Ship(
                        x=random.uniform(0, self.size),
                        y=random.uniform(0, self.size),
                        target_x=random.uniform(0, self.size),
                        target_y=random.uniform(0, self.size),
                        speed=random.uniform(0, MAX_SPEED_RANDOM_SHIP),
                        heading=random.uniform(-180, 180),
                        effective_speed=EFFECTIVE_SPEED_RANDOM_SHIP,
                        max_speed=MAX_SPEED_RANDOM_SHIP
                    )
                )

    def get_nearest_ships(self, ship_idx, count):
        """Get K nearest ships to the given ship"""
        if not (0 <= ship_idx < self.num_ships):
            raise ValueError("Invalid ship index")
        
        if self.ships is None:
            raise ValueError("No ships in the environment")

        ship = self.ships[ship_idx]
        distances = []

        for i, other_ship in enumerate(self.ships):
            if i == ship_idx:
                continue
            distances.append((ship.distance_to(other_ship), i))

        distances.sort(key=lambda item: item[0])
        return [idx for _, idx in distances[:count]]

    def apply_ship_motion(self, ship_idx, delta_speed, delta_heading):
        """Применить изменения скорости и курса к кораблю"""
        if not (0 <= ship_idx < self.num_ships):
            raise ValueError("Invalid ship index")

        if self.ships is None:
            raise ValueError("No ships in the environment")

        ship = self.ships[ship_idx]
        ship.change_speed(delta_speed)
        ship.change_heading(delta_heading)

    def step(self, dt):
        """Execute action for a ship and return next state, reward, done"""
        if self.ships is None:
            raise ValueError("No ships in the environment")

        for ship in self.ships:
            ship.go_straight(dt)
        
        self.time_step += dt

    def get_ship(self, ship_idx: int) -> Ship:
        if not (0 <= ship_idx < self.num_ships):
            raise ValueError("Invalid ship index")

        if self.ships is None:
            raise ValueError("No ships in the environment")

        return self.ships[ship_idx]
    
    def get_nsteps(self):
        return self.time_step