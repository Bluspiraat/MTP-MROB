import heapq
from typing import List, Dict, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from grid import *


def _heuristic(node: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Computes the manhattan distance from the input node to the goal node.

    Args:
        node: The for which the heuristic is to be calculated.
        goal: The goal node location.

    Returns:
        An integer value denoting the manhattan distance to the goal node
    """
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def _reconstruct_path(node: Tuple[int, int], came_from: Dict[Tuple[int, int], Tuple[int, int]]) -> (
        List)[Tuple[int, int]]:
    path: List[Tuple[int, int]] = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return path[::-1]


def _traversal_cost(current_cost, neighbour_cost) -> float:
    """
    Computes the average cost between the two cells

    Args:
        current_cost: The traversability cost of the current cell
        neighbour_cost: The traversability cost of the neighbour cell

    Returns:
        An float value denoting the average cost between the two cells
    """
    return (current_cost + neighbour_cost)/2.0


class AStarOccupancy:
    """
    Implementation of A* search algorithm which finds a path using manhattan distance.
    It works based on an Occupancy Grid.
    """

    def __init__(self, grid: OccupancyGrid):
        self.grid: OccupancyGrid = grid
        self.start = None
        self.goal = None
        self.path = []                                                  # Sequence of nodes which forms the path

    def start_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Performs A* search on a grid given a start and goal coordinates.

        Args:
            start: Is a Tuple[int, int] representing the start coordinate.
            goal: Is a Tuple[int, int] representing the goal coordinate.

        Returns:
            A list of shortest path from the start to the goal coordinates. None if no path exists.
        """
        self.start = start
        self.goal = goal

        open_heap = []
        open_set = set()

        g_scores: Dict[Tuple[int, int], int] = {start: 0}               # Cost to travel to a certain node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}          # The parent node of a give node

        f_start = _heuristic(start, self.goal)
        heapq.heappush(open_heap, (f_start, start))
        open_set.add(start)                                             # Possible nodes to check
        closed_set: Set[Tuple[int, int]] = set()                        # Set of nodes with fully optimized paths

        time_start = time.time()                                        # Track starting time of the search
        while open_heap:
            f_current, current = heapq.heappop(open_heap)               # Retrieves pair with the lowest f-score

            # Skip outdated entries --> Node might be in the closed set, but not popped from the heap
            if current in closed_set:
                continue

            # If current node is the goal node
            if current == goal:
                self.path = _reconstruct_path(current, came_from)
                print(f'Path found, length: {len(self.path)} in {time.time() - time_start}')
                return self.path

            # Move current into closed set
            open_set.remove(current)
            closed_set.add(current)

            # Evaluate all neighbours of current node
            for neighbour in self.grid.get_neighbours(current):

                # If a neighbour is in the closed set, then its value is already optimal
                if neighbour in closed_set:
                    continue

                # The cost to move to the next cell is 1 as it only uses wind directions
                tentative_g = g_scores[current] + 1

                # The old score is the current best path to the neighbour, if it has not been computed before, then the
                # value None is returned
                g_old = g_scores.get(neighbour, None)

                # If tentative_g is larger than the old, nothing should happen as no optimization took place, however,
                # if the node has no associated cost or the cost reduced, then the node should be updated.
                if g_old is None or tentative_g < g_old:
                    g_scores[neighbour] = tentative_g
                    came_from[neighbour] = current

                    f_score: int = tentative_g + _heuristic(neighbour, self.goal)

                    # It might occur that a node exists multiple times in the heap with different f-scores.
                    # Re-processing of a node is avoided by the check if the node is in the closed set after selecting
                    # a new current node
                    heapq.heappush(open_heap, (f_score, neighbour))
                    open_set.add(neighbour)  # Adding does not matter, if it is already in there, it will be overwritten

        print('No path found')
        return None

    def display_path_on_grid(self):
        """
        Displays a white grid with the path in red.
        """
        img = np.ones((self.grid.cost.shape[0], self.grid.cost.shape[1], 3))

        # Draw the path in red
        for (r, c) in self.path:
            img[r, c] = [1, 0, 0]  # red

        plt.figure(figsize=(6, 6))
        plt.title(f'Path length: {len(self.path)}')
        plt.imshow(img)
        plt.show()

    def run_interactive(self):
        """
        Runs an interactive A* search algorithm with the provided occupancy grid.
        Left mouse button sets the start location and the right mouse button the goal location.
        Changing either computes the shortest path from the  start location to the goal location is required.
        """
        matplotlib.use('TkAgg')  # Ensure GUI backend for PyCharm
        fig, ax = plt.subplots()

        def _draw_grid():
            ax.clear()
            ax.imshow(self.grid.occupancy_grid, cmap='Greys')
            if self.start:
                ax.scatter(self.start[1], self.start[0], c='green', s=100, label='Start')
            if self.goal:
                ax.scatter(self.goal[1], self.goal[0], c='red', s=100, label='Goal')
            if self.path:
                px, py = zip(*self.path)
                ax.plot(py, px, c='orange', linewidth=2, label='Path')
            ax.legend()
            fig.canvas.draw()

        def _onclick(event):
            if event.xdata is None or event.ydata is None:
                return  # Ignore clicks outside the axes

            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if event.button == 1:
                self.start = (y, x)
                print(f"Start set at {self.start}")
            elif event.button == 3:
                self.goal = (y, x)
                print(f"Goal set at {self.goal}")

            # Reset previous search
            self.path = []
            self.open_set = []
            self.f_scores = []
            self.closed_set = set()
            self.g_scores = {}
            self.came_from = {}

            if self.start and self.goal:
                self.path = self.start_search(self.start, self.goal)

            _draw_grid()

        fig.canvas.mpl_connect('button_press_event', _onclick)
        _draw_grid()

        # Ensure the window stays open and interactive in PyCharm
        plt.show(block=True)


class AStar:

    def __init__(self, grid: Grid):
        self.grid: Grid = grid
        self.start = None
        self.goal = None
        self.path = []

    def start_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Performs A* search on a grid given a start and goal coordinates.

        Args:
            start: Is a Tuple[int, int] representing the start coordinate.
            goal: Is a Tuple[int, int] representing the goal coordinate.

        Returns:
            A list of shortest path from the start to the goal coordinates. None if no path exists.
        """
        self.start = start
        self.goal = goal
        cost_map = self.grid.cost

        open_heap = []
        open_set = set()

        g_scores: Dict[Tuple[int, int], float] = {start: 0}               # Cost to travel to a certain node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}          # The parent node of a give node

        f_start = _heuristic(start, goal)
        heapq.heappush(open_heap, (f_start, start))
        open_set.add(start)                                             # Possible nodes to check
        closed_set: Set[Tuple[int, int]] = set()                        # Set of nodes with fully optimized paths

        time_start = time.time()                                        # Track starting time of the search
        while open_heap:
            f_current, current = heapq.heappop(open_heap)               # Retrieves pair with the lowest f-score

            # Skip outdated entries --> Node might be in the closed set, but not popped from the heap
            if current in closed_set:
                continue

            # If current node is the goal node
            if current == goal:
                self.path = _reconstruct_path(current, came_from)
                print(f'Path found, computation time: {len(self.path)} in {time.time() - time_start:.2f} seconds, '
                      f'with number of visited nodes: {len(closed_set)} '
                      f'({len(closed_set)/(self.grid.shape[0]*self.grid.shape[1]):.4f}%)')
                return self.path

            # Move current into closed set
            open_set.remove(current)
            closed_set.add(current)

            # Evaluate all neighbours of current node
            for neighbour in self.grid.get_neighbours(current):

                # If a neighbour is in the closed set, then its value is already optimal
                if neighbour in closed_set:
                    continue

                # The cost to move to the next cell is calculated by _traversal_cost
                tentative_g = g_scores[current] + _traversal_cost(cost_map[current], cost_map[neighbour])

                # The old score is the current best path to the neighbour, if it has not been computed before, then the
                # value None is returned
                g_old = g_scores.get(neighbour, None)

                # If tentative_g is larger than the old, nothing should happen as no optimization took place, however,
                # if the node has no associated cost or the cost reduced, then the node should be updated.
                if g_old is None or tentative_g < g_old:
                    g_scores[neighbour] = tentative_g
                    came_from[neighbour] = current

                    f_score: float = tentative_g + _heuristic(neighbour, goal)

                    # It might occur that a node exists multiple times in the heap with different f-scores.
                    # Re-processing of a node is avoided by the check if the node is in the closed set after selecting
                    # a new current node
                    heapq.heappush(open_heap, (f_score, neighbour))
                    open_set.add(neighbour)  # Adding does not matter, if it is already in there, it will be overwritten

        print('No path found')
        return None

    def display_path_on_grid(self):
        """
        Displays a white grid with the path in red.
        """
        color_coded = self.grid.get_colored_grid()

        plt.figure(figsize=(8, 8))
        plt.imshow(color_coded)

        # Draw the path in red
        if self.path:
            path_rows, path_cols = zip(*self.path)
            plt.scatter(path_cols, path_rows, s=3, c="red")

        plt.title(f'Path length: {len(self.path)}')
        plt.show()

    def run_interactive(self):
        """
        Runs an interactive A* search algorithm with the provided occupancy grid.
        Left mouse button sets the start location and the right mouse button the goal location.
        Changing either computes the shortest path from the  start location to the goal location is required.
        """
        matplotlib.use('TkAgg')  # Ensure GUI backend for PyCharm
        fig, ax = plt.subplots()

        def _draw_grid():
            ax.clear()
            ax.imshow(self.grid.get_colored_grid())
            if self.start:
                ax.scatter(self.start[1], self.start[0], c='green', s=100, label='Start')
            if self.goal:
                ax.scatter(self.goal[1], self.goal[0], c='red', s=100, label='Goal')
            if self.path:
                px, py = zip(*self.path)
                ax.plot(py, px, c='red', linewidth=2, label='Path')
            ax.legend()
            fig.canvas.draw()

        def _onclick(event):
            if event.xdata is None or event.ydata is None:
                return  # Ignore clicks outside the axes

            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if event.button == 1:
                self.start = (y, x)
                print(f"Start set at {self.start}")
            elif event.button == 3:
                self.goal = (y, x)
                print(f"Goal set at {self.goal}")

            # Reset previous search
            self.path = []
            self.open_set = []
            self.f_scores = []
            self.closed_set = set()
            self.g_scores = {}
            self.came_from = {}

            if self.start and self.goal:
                self.path = self.start_search(self.start, self.goal)

            _draw_grid()

        fig.canvas.mpl_connect('button_press_event', _onclick)
        _draw_grid()

        # Ensure the window stays open and interactive in PyCharm
        plt.show(block=True)
