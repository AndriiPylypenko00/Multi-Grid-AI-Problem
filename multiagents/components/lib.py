from typing import Tuple
import numpy as np


def calculate_walls(ax: int, ay: int, svet: np.ndarray) -> Tuple[int, ...]:
    return tuple(
        [
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, 0, -1, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, -1, -1, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, -1, 0, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, -1, 1, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, 0, 1, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, 1, 1, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, 1, 0, svet)]),
            calculate_distance(*[ax, ay, *find_wall_pos(ax, ay, 1, -1, svet)]),
        ]
    )


def calculate_cookies(ax: int, ay: int, svet: np.ndarray) -> Tuple[int, ...]:
    return tuple(
        [
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, 0, -1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, -1, -1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, -1, 0, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, -1, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, 0, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, 1, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, 1, 0, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_cookie_pos(ax, ay, 1, -1, svet, ax, ay)]),
        ]
    )


def calculate_agents(ax: int, ay: int, svet: np.ndarray) -> Tuple[int, ...]:
    return tuple(
        [
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, 0, -1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, -1, -1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, -1, 0, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, -1, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, 0, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, 1, 1, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, 1, 0, svet, ax, ay)]),
            calculate_distance(*[ax, ay, *find_agent_pos(ax, ay, 1, -1, svet, ax, ay)]),
        ]
    )


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return ((x2 - x1) ** 2) + ((y2 - y1) ** 2)


def find_cookie_pos(ax: int, ay: int, x: int, y: int, svet: np.ndarray, rx: int, ry: int) -> Tuple[int, int]:
    try:
        assert ax + x > 0 and ay + y > 0
        if svet[ax + x, ay + y] == 1:
            return (ax + x, ay + y)
        else:
            return find_cookie_pos(ax + x, ay + y, x, y, svet, rx, ry)
    except (IndexError, AssertionError):
        return (rx, ry)


def find_agent_pos(ax: int, ay: int, x: int, y: int, svet: np.ndarray, rx: int, ry: int) -> Tuple[int, int]:
    try:
        assert ax + x > 0 and ay + y > 0
        if svet[ax + x, ay + y] > 3:
            return (ax + x, ay + y)
        else:
            return find_agent_pos(ax + x, ay + y, x, y, svet, rx, ry)
    except (IndexError, AssertionError):
        return (rx, ry)


def find_wall_pos(ax: int, ay: int, x: int, y: int, svet: np.ndarray) -> Tuple[int, int]:
    if svet[ax + x, ay + y] == -1:
        return (ax + x, ay + y)
    else:
        return find_wall_pos(ax + x, ay + y, x, y, svet)
