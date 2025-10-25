import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from maritime_env import Ship
from maritime_env import MaritimeEnvironment
from ddqn_model import DQN
import rl_wrapper_maritime_env
import torch
import json
import os
import matplotlib.patches as patches
from matplotlib.patches import Circle
import rl_config

class MaritimeVisualizer:
    def __init__(self, env, ship_idx=0):
        self.env = env
        self.ship_idx = ship_idx
        self.ship_size = 5
        self.safety_radius = 50

        # Создаем фигуру
        self.fig, self.ax = plt.subplots()
        self.ship_points, = self.ax.plot([], [], 'bo', markersize=self.ship_size, label='Ships')
        self.target_points, = self.ax.plot([], [], 'rx', label='Targets')
        self.main_ship_point, = self.ax.plot([], [], 'go', markersize=self.ship_size, label='Main Ship')
        self.heading_arrows = []

        # Камера
        self.xlim = None
        self.ylim = None

        self._init_safety_zones()


    # Инициализация зон безопасности
    def _init_safety_zones(self):
        self.safety_zones = []
        for ship in self.env.ships:
            r = self.safety_radius
            circle = Circle((ship.get_state()['x'], ship.get_state()['y']), r, color='blue', alpha=0.1)
            self.ax.add_patch(circle)
            self.safety_zones.append(circle)

    def init_plot(self):
        """Инициализация графика"""
        self.ship_points.set_data([], [])
        self.target_points.set_data([], [])
        self.main_ship_point.set_data([], [])
        self.heading_arrows = []

        # Устанавливаем начальные границы камеры по главному кораблю
        main_ship = self.env.get_ship(self.ship_idx)
        camera_range = self.env.size
        self.xlim = (main_ship.get_state()['x'] - camera_range, main_ship.get_state()['x'] + camera_range)
        self.ylim = (main_ship.get_state()['y'] - camera_range, main_ship.get_state()['y'] + camera_range)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)

        return self.ship_points, self.target_points, self.main_ship_point

    def _draw_arrows(self):
        """Отображение направления движения"""
        # Удаляем старые стрелки
        for arrow in self.heading_arrows:
            arrow.remove()
        self.heading_arrows.clear()

        for ship in self.env.ships:
            x, y = ship.get_state()['x'], ship.get_state()['y']
            angle = math.radians(ship['heading'])
            dx = math.cos(angle) * 20
            dy = math.sin(angle) * 20
            arrow = self.ax.arrow(x, y, dx, dy, head_width=10, head_length=15, fc='blue', ec='blue')
            self.heading_arrows.append(arrow)

    def update_camera(self, main_ship):
        """Плавная камера с deadzone"""
        deadzone = self.env.size // 5
        smooth_factor = 1

        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        dx = dy = 0

        # Проверяем X
        if main_ship.get_state()['x'] < x_min + deadzone:
            dx = (main_ship.get_state()['x'] - (x_min + deadzone)) * smooth_factor
        elif main_ship.get_state()['x'] > x_max - deadzone:
            dx = (main_ship.get_state()['x'] - (x_max - deadzone)) * smooth_factor

        # Проверяем Y
        if main_ship.get_state()['y'] < y_min + deadzone:
            dy = (main_ship.get_state()['y'] - (y_min + deadzone)) * smooth_factor
        elif main_ship.get_state()['y'] > y_max - deadzone:
            dy = (main_ship.get_state()['y'] - (y_max - deadzone)) * smooth_factor

        # Обновляем границы
        self.xlim = (x_min + dx, x_max + dx)
        self.ylim = (y_min + dy, y_max + dy)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)

    def _update_safety_zones(self):
        for ship, circle in zip(self.env.ships, self.safety_zones):
            circle.center = (ship.get_state()['x'], ship.get_state()['y'])
            circle.set_facecolor((0, 0, 1, 0.1))  # базовый цвет

        # Проверка пересечений
        n = self.env.num_ships
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.env.get_ship(i).get_state()['x'] - self.env.get_ship(j).get_state(j)['x']
                dx = self.env.get_ship(i).get_state()['y'] - self.env.get_ship(j).get_state(j)['y']
                dist = (dx**2 + dy**2)**0.5
                r1 = self.safety_radius
                r2 = self.safety_radius
                if dist < r1 + r2:
                    self.safety_zones[i].set_facecolor((1, 0, 0, 0.2))
                    self.safety_zones[j].set_facecolor((1, 0, 0, 0.2))

    def _draw_ships(self):
        xs = [ship.get_state()['x'] for ship in self.env.ships]
        ys = [ship.get_state()['y'] for ship in self.env.ships]
        self.ship_points.set_data(xs, ys)

        main_ship = self.env.get_ship(self.ship_idx)
        self.main_ship_point.set_data([main_ship.get_state()['x']], [main_ship.get_state()['y']])

        txs = [self.env.get_ship(self.ship_idx).get_state()['target_x']]
        tys = [self.env.get_ship(self.ship_idx).get_state()['target_y']]
        self.target_points.set_data(txs, tys)

    def update(self, frame):
        main_ship = self.env.get_ship(self.ship_idx)

        self._draw_ships()
        self._update_safety_zones()
        self._draw_arrows()
        self.update_camera(main_ship)

        # Возвращаем объекты для FuncAnimation
        return (
            self.ship_points, 
            self.main_ship_point, 
            self.target_points, 
            *self.heading_arrows, 
            *self.safety_zones
        )

    def run(self, step_callback=None, interval=100, steps=1000, save_path=None, fps=30):
        """
        Запуск анимации.

        :param step_callback: функция обновления состояния среды
        :param interval: интервал между кадрами (мс)
        :param steps: количество кадров
        :param save_path: путь для сохранения видео (например, 'simulation.mp4')
        :param fps: количество кадров в секунду при сохранении видео
        """
        def animate(frame):
            if step_callback:
                step_callback(frame)
            return self.update(frame)

        # Создаем анимацию и сохраняем в объекте, чтобы он не удалился
        self.anim = animation.FuncAnimation(
            self.fig,
            animate,
            init_func=self.init_plot,
            frames=steps,
            interval=interval,
            blit=False
        )

        # Если указан путь для сохранения, сохраняем видео
        if save_path is not None:
            try:
                self.anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'], )
                print(f"Видео успешно сохранено: {save_path}")
            except Exception as e:
                print(f"Ошибка при сохранении видео: {e}")

        # Пока не показываем анимацию
        # plt.show()


class GeneratorVideo:
    """
    Класс для гибкой генерации видео морских симуляций.
    Позволяет повторно использовать один объект для разных сред и моделей.
    """

    def __init__(self, 
                 save_dir: str = "videos"):
        self.save_dir  = save_dir

        # Инициализация среды и модели пустыми
        self.ship_idx = 0
        self.rl_env = None
        self.env = None
        self.agent = None
        self.visualizer = None
        self.model_name = None

        # Гарантируем, что папка для видео существует
        os.makedirs(self.save_dir, exist_ok=True)

    def load_environment(self, env_name: str):
        """
        Загружает состояние среды из JSON-шаблона.

        :param env_name: имя шаблона 
        """
        print(f"[INFO] Загружаем окружение: {env_name}")

        # Создаем новую среду

        # Путь к шаблону
        template_path = os.path.join("environments", env_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Шаблон не найден: {template_path}")
        if not env_name.endswith(".json"):
            raise FileNotFoundError("File is not json")

        # Читаем данные шаблона
        with open(template_path, "r") as f:
            data = json.load(f)

        ships_data = data.get("ships", [])

        ships = []

        # Очищаем текущие корабли
        num_ships = len(ships_data)

        # Загружаем корабли из шаблона
        for ship_info in ships_data:
            ship = Ship(
                ship_info.get('x', 0),
                ship_info.get('y', 0),
                ship_info.get('target_x', 0),
                ship_info.get('target_y', 0),
                ship_info.get('speed', 0),
                ship_info.get('heading', 0),
                ship_info.get('effective_speed', 5),
                ship_info.get('max_speed', 7)
            )
            ships.append(ship)

        self.rl_env = rl_wrapper_maritime_env.RLWrapperMaritimeEnv(rl_config.MAP_SIZE, rl_config.MAX_STEPS, rl_config.NUMBER_OF_NEAREST_SHIPS, num_ships, ships, self.ship_idx)

        self.env = self.rl_env.env

        print(f"[OK] Среда успешно загружена: {self.env.num_ships} кораблей")

    def load_model(self, model_name: str):
        """Загружает обученную модель агента."""
        print(f"[INFO] Загружаем модель: {model_name}")
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model_name = model_name

         # Calculate state and action dimensions
        # State: [dx, dy, speed, heading] + k_nearest * [distance, rel_speed, rel_heading]
        state_dim = 4 + rl_config.NUMBER_OF_NEAREST_SHIPS * 3
        # Action: speed_changes (7) * heading_changes (7)
        action_dim = len(rl_config.SHIP_SPEED_CHANGES) * len(rl_config.SHIP_HEADING_CHANGES)

        # Создаем нового агента под текущую среду
        self.agent = DQN(
            state_dim=state_dim,
            action_dim=action_dim
        )

        self.agent.load_state_dict(torch.load(model_path))
        self.agent.eval()

        print("[OK] Модель успешно загружена.")

    def _prepare_visualizer(self):
        """Создает визуализатор для текущей среды."""
        if self.env is None:
            raise ValueError("Окружение не загружено. Используй load_environment().")
        self.visualizer = MaritimeVisualizer(self.env, ship_idx=self.ship_idx)

    def _step_callback(self, frame):
        """Функция шага симуляции."""
        state = torch.FloatTensor(self.rl_env._get_state(self.ship_idx))
        action = self.agent(state)
        self.rl_env.step(action)

    def run_simulation(self, steps: int = 500, interval: int = 100, fps: int = 10, output_name: str = None):
        """
        Запускает симуляцию и сохраняет видео.
        Можно вызывать много раз с разными шаблонами и моделями.
        """
        if self.env is None:
            raise ValueError("Окружение не загружено. Используй load_environment().")
        if self.agent is None:
            raise ValueError("Модель не загружена. Используй load_model().")
        if self.rl_env is None:
            raise ValueError("Обертка на среду не загрузилась ( если сюда дошло то гг :) )")

        self._prepare_visualizer()

        # Путь для сохранения
        if output_name is None:
            if self.model_name:
                output_name = self.model_name.replace(".pth", ".mp4")
            else:
                output_name = "simulation.mp4"
        save_path = os.path.join(self.save_dir, output_name)

        print(f"[INFO] Запуск симуляции ({steps} шагов)...")

        self.visualizer.run(
            step_callback=self._step_callback,
            interval=interval,
            steps=steps,
            save_path=save_path,
            fps=fps
        )

        print(f"[SUCCESS] Видео сохранено: {save_path}")



if __name__ == "__main__":
    generator = GeneratorVideo(save_dir="videos")

    # Загружаем окружение и модель
    generator.load_environment("env1.json")
    generator.load_model("ship_collision_avoidance_model320.pth")

    # Генерируем видео
    generator.run_simulation(steps=400, fps=10, output_name="env1_model320.mp4")

