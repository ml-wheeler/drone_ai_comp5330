import matplotlib.pyplot as plt
import numpy as np
import heapq

class Receiver:
    def __init__(self, x, y, sensitivity):
        self.x = x
        self.y = y
        self.sensitivity = sensitivity


class Drone:
    def __init__(self, x = 0, y = 0, speed = 10, angle = 0, start=(0, 0)):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle
        self.noise = 0.3 * speed + 0.35 * np.abs(np.sin(angle))
        self.start = start


class Space:
    def __init__(self, x, y, receivers=None, drones=None):
        self.x = x
        self.y = y
        self.grid = None
        self.drones = drones
        self.receivers = receivers

    def build_grid(self):
        self.grid = [
            (x_i, y_i)
            for x_i in range(self.x + 1)
            for y_i in range(self.y + 1)
        ]


def calc_noise_penalty(x, y, drone, receiver_list, k=5, epsilon=1.0):
    max_pen = 0

    for r in receiver_list:
        s = r.sensitivity

        distance_squared = (x - r.x)**2 + (y - r.y)**2

        penalty = (
            (np.exp(k * s) - 1) * drone.noise
        ) / (distance_squared + epsilon)

        max_pen = max(max_pen, penalty)

    return max_pen


def euclid_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def path_distance(path):
    if path is None or len(path) < 2:
        return 0.0
    return sum(euclid_dist(a, b) for a, b in zip(path, path[1:]))

# Return cumulative noise cost

def generate_path(drone, goal, receiver_list, space, k=5, noise_weight=1.0, epsilon=1.0):
    start = tuple(drone.start)
    goal = tuple(goal)
    noise_weight = float(noise_weight)
    if not 0.0 <= noise_weight <= 1.0:
        raise ValueError("noise_weight must be between 0 and 1")

    if not (0 <= start[0] <= space.x and 0 <= start[1] <= space.y):
        return None, None
    if not (0 <= goal[0] <= space.x and 0 <= goal[1] <= space.y):
        return None, None

    distance_weight = 1.0 - noise_weight
    movements = (
        (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0),                     (0, 1, 1.0),
        (1, -1, np.sqrt(2)),  (1, 0, 1.0), (1, 1, np.sqrt(2)),
    )

    def heuristic(position):
        return distance_weight * euclid_dist(position, goal)

    open_set = [(heuristic(start), 0.0, start)]
    g_score = {start: 0.0}
    came_from = {}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current_g != g_score.get(current):
            continue

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            cumulative_noise = sum(
                calc_noise_penalty(x, y, drone, receiver_list, k=k, epsilon=epsilon)
                for x, y in path
            )
            return cumulative_noise, path

        for dx, dy, distance in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] <= space.x and 0 <= neighbor[1] <= space.y):
                continue

            noise_penalty = calc_noise_penalty(
                neighbor[0], neighbor[1], drone, receiver_list, k=k, epsilon=epsilon
            )
            step_cost = distance_weight * distance + noise_weight * noise_penalty
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    open_set,
                    (tentative_g + heuristic(neighbor), tentative_g, neighbor),
                )

    return None, None


from matplotlib.widgets import Slider, TextBox, Button
import ast
from matplotlib.lines import Line2D
proxy = Line2D([], [], marker='o', color='red', linestyle='none', label='Receivers')

def create_dashboard(drone, space, max_receivers=6):
    lambda_val = 0.5
    k_val = 3
    goal = (100, 100)
    receiver_list = []
    recv_widgets = []

    fig, ax = plt.subplots(figsize=(14, 7))

    plt.subplots_adjust(left=0.5)
    initial_noise, path = generate_path(
        drone, goal, receiver_list, space, k=k_val, noise_weight=lambda_val
    )

    path = np.array(path)

    ax.grid(alpha=0.5, mouseover=True, zorder=1.0)
    ax.set_xticks(np.arange(0, 101, 5))
    ax.set_xlabel('x')
    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_ylabel('y')
    l, = ax.plot(path[:, 0], path[:, 1], lw=2)

    start_pt, = ax.plot([drone.start[0]], [drone.start[1]], marker='o', color='green', linestyle='none', label='Start', zorder=10)
    goal_pt, = ax.plot([goal[0]], [goal[1]], marker='o', color='blue', linestyle='none', label='Goal', zorder=10)

    recv_sc = ax.scatter([], [], c='red', zorder=10, label='Receivers')
    ax.legend(loc='upper left')

    score_txt = fig.text(0.04, 0.26, f'Noise score: {initial_noise:.2f}', fontsize=11,
                         fontweight='bold', va='center')
    distance_txt = fig.text(0.25, 0.26, f'Distance: {path_distance(path):.2f}', fontsize=11,
                            fontweight='bold', va='center')

    for r in receiver_list:
        plt.plot(r.x, r.y, marker='o', markersize=r.sensitivity*10, color='red', zorder=10.0)

    ax_slider = plt.axes([0.08, 0.90, 0.14, 0.05], facecolor='blue')
    slider = Slider(ax_slider, label='Lambda', valmin=0.0, valmax=1.0, valinit=0.5, valstep=0.05)

    ax_kslider = plt.axes([0.28, 0.90, 0.14, 0.05], facecolor='blue')
    kslider = Slider(ax_kslider, label='k', valmin=1, valmax=5, valinit=3, valstep=1)

    ax_start = plt.axes([0.09, 0.20, 0.12, 0.04])
    ax_goal = plt.axes([0.28, 0.20, 0.12, 0.04])
    ax_speed = plt.axes([0.09, 0.10, 0.12, 0.04])

    start_box = TextBox(ax_start, initial=f'{drone.start}', label='Drone start')
    goal_box = TextBox(ax_goal, initial=f'{goal}', label='Goal')
    speed_box = TextBox(ax_speed, initial=f'{drone.speed}', label='Drone speed')

    ax_addrecv = plt.axes([0.04, 0.80, 0.20, 0.05])
    ax_resrecv = plt.axes([0.25, 0.80, 0.20, 0.05])
    addrecv_btn = Button(ax=ax_addrecv, label='Add receiver')
    resrecv_btn = Button(ax=ax_resrecv, label='Reset receivers')


    for i in range(max_receivers):
        y0 = 0.71 - i*0.08
        h = 0.05

        # col1
        ax_num = plt.axes([0.04, y0, 0.04, h])
        ax_num.axis('off')
        ax_num.text(0.5, 0.5, str(i+1), ha='center', va='center')

        # col2
        ax_x = plt.axes([0.09, y0, 0.05, h])
        x_slider = Slider(ax_x, label='x', valmin=0.0, valmax=100.0,
                          valinit=50, valstep=0.1)

        ax_y = plt.axes([0.17, y0, 0.05, h])
        y_slider = Slider(ax_y, label='y', valmin=0.0, valmax=100.0,
                          valinit=50, valstep=0.1)

        # col3
        ax_sens = plt.axes([0.26, y0, 0.10, h])
        sens = Slider(ax_sens, label='', valmin=0.0, valmax=1.0,
                      valinit=0.5, valstep=0.05)

        # col4
        ax_del = plt.axes([0.39, y0, 0.06, h])
        del_btn = Button(ax_del, label='Remove')
        recv_widgets.append({
            'axes': [ax_num, ax_x, ax_y, ax_sens, ax_del],
            'x': x_slider, 'y': y_slider, 'sens': sens, 'del':del_btn,
        })

        x_slider.on_changed(lambda text, i=i: on_x(i, text))
        y_slider.on_changed(lambda text, i=i: on_y(i, text))
        sens.on_changed(lambda val, i=i: on_sens(i, val))
        del_btn.on_clicked(lambda event, i=i: on_del(i))


    def sync_rows():
        for i, row in enumerate(recv_widgets):
            shown = i < len(receiver_list)

            for ax_ in row['axes']:
                ax_.set_visible(shown)
            for w in (row['x'], row['y'], row['sens'], row['del']):
                w.set_active(shown)

            if shown:
                r=receiver_list[i]
                row['x'].eventson = row['y'].eventson = row['sens'].eventson = False
                row['x'].set_val(r.x)
                row['y'].set_val(r.y)
                row['sens'].set_val(r.sensitivity)
                row['x'].eventson = row['y'].eventson = row['sens'].eventson = True

        fig.canvas.draw_idle()


    def add_recv(event):
        if len(receiver_list) >= max_receivers:
            return
        receiver_list.append(Receiver(50, 50, 0.5))
        sync_rows()
        update()


    def on_x(i, val):
        try:
            x = val
        except (ValueError, SyntaxError):
            return
        receiver_list[i].x = x
        update()

    def on_y(i, val):
            try:
                y = val
            except (ValueError, SyntaxError):
                return
            receiver_list[i].y = y
            update()

    def on_sens(i, val):
        receiver_list[i].sensitivity = val
        update()


    def on_del(i):
        receiver_list.pop(i)
        sync_rows()
        update()


    def on_reset(event):
        receiver_list.clear()
        sync_rows()
        update()


    def update(_=None):
        start_pt.set_data([drone.start[0]], [drone.start[1]])
        goal_pt.set_data([goal[0]], [goal[1]])

        if receiver_list:
            recv_sc.set_offsets([[r.x, r.y] for r in receiver_list])
            recv_sc.set_sizes([max(5 * r.sensitivity, 1) ** 2 for r in receiver_list])
        else:
            recv_sc.set_offsets(np.empty((0, 2)))

        total_noise_gen, new_path = generate_path(drone, goal, receiver_list, space,
                                k=int(kslider.val), noise_weight=slider.val)
        
        if new_path is not None:
            new_path = np.array(new_path)
            l.set_data(new_path[:, 0], new_path[:, 1])
            score_txt.set_text(f'Noise score: {total_noise_gen:.2f}')
            distance_txt.set_text(f'Distance: {path_distance(new_path):.2f}')
        else:
            score_txt.set_text('Noise score: no path found')
            distance_txt.set_text('Distance: --')

        fig.canvas.draw_idle()


    def on_start(text):
        drone.start = ast.literal_eval(text)
        update()


    def on_goal(text):
        nonlocal goal
        goal = ast.literal_eval(text)
        update()


    def on_speed(text):
        drone.speed = float(text)
        update()


    def on_reset(event):
        receiver_list.clear()
        sync_rows()
        update()


    slider.on_changed(update)
    kslider.on_changed(update)
    start_box.on_submit(on_start)
    goal_box.on_submit(on_goal)
    speed_box.on_submit(on_speed)

    addrecv_btn.on_clicked(add_recv)
    resrecv_btn.on_clicked(on_reset)

    sync_rows()
    plt.show()


space = Space(100, 100)
drone = Drone(x=0, y=0, speed=10)


create_dashboard(drone, space)
