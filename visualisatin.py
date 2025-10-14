import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from main import MaritimeEnvironment
from main import MultiShipCollisionAvoidance
import torch
import json
import os

class MaritimeVisualizer:
    def __init__(self, env, ship_idx=0):
        self.env = env
        self.ship_idx = ship_idx
        self.ship_size = 5

        # Создаем фигуру
        self.fig, self.ax = plt.subplots()
        self.ship_points, = self.ax.plot([], [], 'bo', markersize=self.ship_size, label='Ships')
        self.target_points, = self.ax.plot([], [], 'rx', label='Targets')
        self.main_ship_point, = self.ax.plot([], [], 'go', markersize=self.ship_size, label='Main Ship')
        self.heading_arrows = []

        # Камера
        self.xlim = None
        self.ylim = None

    def init_plot(self):
        """Инициализация графика"""
        self.ship_points.set_data([], [])
        self.target_points.set_data([], [])
        self.main_ship_point.set_data([], [])
        self.heading_arrows = []

        # Устанавливаем начальные границы камеры по главному кораблю
        main_ship = self.env.ships[self.ship_idx]
        camera_range = self.env.size
        self.xlim = (main_ship['x'] - camera_range, main_ship['x'] + camera_range)
        self.ylim = (main_ship['y'] - camera_range, main_ship['y'] + camera_range)
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
            x, y = ship['x'], ship['y']
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
        if main_ship['x'] < x_min + deadzone:
            dx = (main_ship['x'] - (x_min + deadzone)) * smooth_factor
        elif main_ship['x'] > x_max - deadzone:
            dx = (main_ship['x'] - (x_max - deadzone)) * smooth_factor

        # Проверяем Y
        if main_ship['y'] < y_min + deadzone:
            dy = (main_ship['y'] - (y_min + deadzone)) * smooth_factor
        elif main_ship['y'] > y_max - deadzone:
            dy = (main_ship['y'] - (y_max - deadzone)) * smooth_factor

        # Обновляем границы
        self.xlim = (x_min + dx, x_max + dx)
        self.ylim = (y_min + dy, y_max + dy)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)

    def update(self, frame):
        """Обновление кадра"""
        main_ship = self.env.ships[self.ship_idx]
        self.update_camera(main_ship)

        xs = [ship['x'] for ship in self.env.ships]
        ys = [ship['y'] for ship in self.env.ships]

        #txs = [ship['target_x'] for ship in self.env.ships]
        #tys = [ship['target_y'] for ship in self.env.ships]

        txs = [env.ships[0]['target_x']]
        tys = [env.ships[0]['target_y']]

        self.ship_points.set_data(xs, ys)
        self.target_points.set_data(txs, tys)

        self.main_ship_point.set_data([main_ship['x']], [main_ship['y']])

        # Обновляем стрелки направления
        self._draw_arrows()


        return self.ship_points, self.target_points, self.main_ship_point, *self.heading_arrows

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

        # Показываем анимацию
        plt.show()



def load_environment_template(env, template_name):
    """
    Загружает состояние среды из файла шаблона.
    
    :param env: объект MaritimeEnvironment
    :param template_name: имя шаблона без расширения (например, 'template1')
    """
    template_path = os.path.join("environments", template_name + ".json")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Шаблон не найден: {template_path}")
    
    with open(template_path, "r") as f:
        data = json.load(f)
    
    ships_data = data.get("ships", [])
    
    # Сбрасываем текущую среду
    env.ships = []
    
    for ship_info in ships_data:
        ship = {
            'x': ship_info.get('x', 0),
            'y': ship_info.get('y', 0),
            'target_x': ship_info.get('target_x', 0),
            'target_y': ship_info.get('target_y', 0),
            'speed': ship_info.get('speed', 0),
            'heading': ship_info.get('heading', 0),
            'desired_speed': ship_info.get('desired_speed', 7)
        }
        env.ships.append(ship)
        print(ship)
    
    # Сбрасываем счётчик времени
    env.time_step = 0




if __name__ == "__main__":
    env = MaritimeEnvironment(num_ships=2, k_nearest=5)
    env_name = "env1"
    load_environment_template(env, env_name)

    # Загрузим обученную модель
    agent = MultiShipCollisionAvoidance(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

    model = "v1_ship_collision_avoidance_model140.pth"

    agent.target_net.load_state_dict(torch.load("models/" + model))
    agent.target_net.eval()
    agent.policy_net.load_state_dict(torch.load("models/" + model))
    agent.policy_net.eval()

    # создаем визуализатор
    visualizer = MaritimeVisualizer(env, ship_idx=0)

    # определим callback-функцию для управления кораблем
    def step_callback(frame):
        state = env._get_state(0)
        action = agent.select_action(state, epsilon=0)
        env.step(0, action)  # обновляем состояние среды

    # Запускаем анимацию
    visualizer.run(step_callback=step_callback, interval=100, steps=500, save_path="videos/" + model[:-4] + ".mp4")

