from typing import List, Dict, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time


class AStarOccupancy:
    def __init__(self, grid):
        self.grid = grid
        self.start = None
        self.goal = None
        self.path = []                                                  # Sequence of nodes which forms the path

    def start_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        self.start = start
        self.goal = goal

        # --- Setup score tracking and node lists --- #
        open_set = [start]                                              # Possible nodes to check
        closed_set: Set[Tuple[int, int]] = set()                        # Set of nodes with fully optimized paths
        f_scores = [self._heuristic(start)]                             # The priority value of nodes
        g_scores: Dict[Tuple[int, int], int] = {start: 0}               # Cost to travel to a certain node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}          # The parent node of a give node

        time_start = time.time()                                        # Track starting time of the search
        while open_set:
            idx = int(np.argmin(f_scores))                              # Pick value with the lowest priority score
            current = open_set[idx]

            # If current node is the goal node
            if current == goal:
                self.path = self._reconstruct_path(current, came_from)
                print(f'Path found, length: {len(self.path)} in {time.time() - time_start}')
                return self.path

            # Move current into closed set
            closed_set.add(current)
            open_set.pop(idx)
            f_scores.pop(idx)

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
                    f_score: int = tentative_g + self._heuristic(neighbour)
                    came_from[neighbour] = current

                    # Situation: The node has not been seen before and is assigned a priority value based on the cost to
                    # get to the node and the heuristic
                    if neighbour not in open_set:
                        open_set.append(neighbour)
                        f_scores.append(f_score)
                    # Situation: The existence of the node is known and its priority value should be updated.
                    else:
                        neighbour_index = list.index(open_set, neighbour)
                        f_scores[neighbour_index] = f_score

        print('No path found')
        return None

    def _heuristic(self, node) -> int:
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def _reconstruct_path(self, node: Tuple[int, int], came_from: Dict[Tuple[int, int], Tuple[int, int]]) -> (
            List)[Tuple[int, int]]:
        path: List[Tuple[int, int]] = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        return path[::-1]

    def display_path_on_grid(self):
        img = np.ones((self.grid.cost.shape[0], self.grid.cost.shape[1], 3))

        # Draw the path in red
        for (r, c) in self.path:
            img[r, c] = [1, 0, 0]  # red

        plt.figure(figsize=(6, 6))
        plt.title(f'Path length: {len(self.path)}')
        plt.imshow(img)
        plt.show()

    def run_interactive(self):
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

