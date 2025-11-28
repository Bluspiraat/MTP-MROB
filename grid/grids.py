import rasterio
import numpy as np
from typing import Dict, Tuple, List
from numpy.typing import NDArray
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
import copy
from tqdm import tqdm


@dataclass
class Grid:
    """
    Represents a segmented terrain grid loaded from a file, along with associated
    cost/traversability mapping.

    Attributes:
        grid_file (str): Path to the raster file containing the raw category indices.
        cost_mapping (List[int]): The 14-element list defining the traversability cost
                                   for each category index (0-13).
        category (NDArray): The calculated 2D array of category indices (uint8).
        cost (NDArray): The calculated 2D array of traversability costs (uint8).
    """
    grid_file: str
    cost_mapping: List[int]
    shape: Tuple[int, int] = field(init=False)

    # --- Standard values --- #
    categories = ["unknown", "hardened", "half-hardened", "unhardened", "track", "fallow", "agriculture",
                  "forest", "grassland", "heath", "dune", "sand", "water", "other"]
    colors = ["white", "black", "dimgray", "darkgray", "darkmagenta", "darkred", "orange", "forestgreen",
              "lightgreen", "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"]

    # --- Main maps --- #
    category: NDArray = field(init=False)
    cost: NDArray = field(init=False)
    resolution: float = field(init=False)

    def __post_init__(self):
        assert len(self.cost_mapping) == 14
        self.category, self.cost = self._load_grid(self.cost_mapping)
        self.shape = self.category.shape
        print(f'Grid loaded with dimension: {self.shape}')

    def _load_grid(self, cost_mapping: List) -> Tuple[NDArray, NDArray]:
        """
        Creates two grids by loading the category data and mapping it to traversal costs.

        The first grid represents the terrain category in each cell, and the second
        represents the cost of traversing that cell.

        Args:
            cost_mapping (List[int]): A list of integers where the index corresponds
                to the terrain category (0-13) and the value is the traversal cost.

        Returns:
            Tuple[NDArray, NDArray]: A tuple containing:
                - NDArray: The category grid (indices 0-13).
                - NDArray: The calculated cost grid.
        """
        category_grid = self._load_data()
        cost_grid = np.empty(category_grid.shape, dtype='uint8')
        for index, value in np.ndenumerate(category_grid):
            cost_grid[index] = cost_mapping[value]
        return category_grid, cost_grid

    def _load_data(self) -> np.ndarray:
        """
        Loads the data from the grid file.

        Returns:
            np.ndarray: The grid data in 'uint8' format.
        """
        with rasterio.open(self.grid_file) as src:
            self.resolution = src.res[0]
            return src.read(1).astype('uint8')

    def display_grid(self):
        """
        Displays the grid in a matplotlib figure.

        """
        file_name = os.path.basename(self.grid_file)
        plt.title(f'Index of segmented tile: {os.path.splitext(file_name)[0]}\n'
                  f'Resolution of grid: {self.resolution}')
        plt.imshow(self.get_colored_grid())
        plt.show()
        return plt

    def get_colored_grid(self):
        """
        Returns a colored grid based on the grid data.

        Returns:
            A matplotlib figure with grid data for visualization
        """
        color_codes = np.array([mcolors.to_rgb(c) for c in self.colors])
        color_coded = color_codes[self.category]
        return color_coded

    def reduce_grid(self, new_origin: Tuple[int, int], size: Tuple[int, int]):
        """
        Reduces the grid to a new grid size, thus creating a subset

        Args:
            new_origin (Tuple[int, int]): The starting row and column of the subset
            size: The size of the grid given in row and columns numbers
        """
        self.cost = self.cost[new_origin[0]:new_origin[0] + size[0], new_origin[1]:new_origin[1] + size[1]]
        self.category = self.category[new_origin[0]:new_origin[0] + size[0], new_origin[1]:new_origin[1] + size[1]]
        self.shape = self.category.shape
        print(f'Grid reduced to size: {self.shape}')

    def get_neighbours(self, location: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        Check the neighbours in all four perpendicular directions and returns if they are accessible.

        Args:
            location Tuple[int, int]: The row and column of the cell of which the neighbours are te be returned.

        Returns:
            neighbours Tuple[Tuple[int, int], ...]: The neighbours of the cell at the given location.
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbours = []
        for row_dir, col_dir in directions:
            row_nb, col_nb = location[0] + row_dir, location[1] + col_dir
            if self.shape[0] - 1 >= row_nb >= 0 and self.shape[1] - 1 >= col_nb >= 0:
                if self.category[row_nb][col_nb] != 0:
                    neighbours.append((row_nb, col_nb))
        return tuple(neighbours)

    def resample(self, reduction: int):
        assert reduction % 2 == 0

        # --- Creat copy and set initial parameters --- #
        resampled_grid = copy.deepcopy(self)
        resampled_grid.resolution = self.resolution * reduction
        resampled_grid.shape = (int(self.shape[0] / reduction), int(self.shape[1] / reduction))

        # --- Resample the categories --- #
        resampled_grid.category = np.zeros(shape=resampled_grid.shape, dtype='uint8')
        resampled_grid.cost = np.empty(resampled_grid.category.shape, dtype='uint8')
        for index, value in tqdm(np.ndenumerate(resampled_grid.category), total=resampled_grid.category.size):
        # for index, value in np.ndenumerate(resampled_grid.category):
            relevant_cells = self.category[index[0] * reduction:index[0] * reduction + reduction,
                                           index[1] * reduction:index[1] * reduction + reduction]
            category = np.argmax(np.bincount(relevant_cells.flatten())).astype(np.uint8)
            resampled_grid.category[index] = category
            resampled_grid.cost[index] = resampled_grid.cost_mapping[category]
        return resampled_grid


@dataclass
class OccupancyGrid:
    """
    Represents an existing Grid object as a binary occupancy grid.

    Attributes:
        base_grid (Grid): A Grid object.
        blocked_categories (List[str]):     A list of strings defining the categories representing blocked cells.
        occupancy_grid (NDArray[np_bool]):  A 2D array of occupancy, free cells are represented by False values,
                                            blocked cells are represented by True values.
    """

    base_grid: Grid
    blocked_categories: List[int]

    occupancy_grid: NDArray[np.bool_] = field(init=False)
    cost: NDArray = field(init=False)
    shape: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        """
        Converts the existing grid in the Grid object into a binary occupancy grid.

        Returns:
            occupancy_grid (np.ndarray[bool]): A 2D array of occupancy.
        """
        lookup = np.zeros(14, dtype=bool)
        lookup[self.blocked_categories] = True
        self.occupancy_grid = lookup[self.base_grid.category]
        self.cost: NDArray[np.int8] = np.ones_like(self.occupancy_grid)
        self.shape = self.cost.shape

    def get_neighbours(self, location: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        Check the neighbours in all four perpendicular directions and returns if they are accessible.

        Args:
            location Tuple[int, int]: The row and column of the cell of which the neighbours are te be returned.

        Returns:
            neighbours Tuple[Tuple[int, int], ...]: The neighbours of the cell at the given location.
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbours = []
        for row_dir, col_dir in directions:
            row_nb, col_nb = location[0] + row_dir, location[1] + col_dir
            if self.shape[0] - 1 >= row_nb >= 0 and self.shape[1] - 1 >= col_nb >= 0 and not \
            self.occupancy_grid[row_nb][col_nb]:
                neighbours.append((row_nb, col_nb))
        return tuple(neighbours)

    def display_grid(self):
        blocked_categories = [self.base_grid.categories[index] for index in self.blocked_categories]
        plt.title(f'Occupancy grid with blocked categories: {blocked_categories}')
        plt.imshow(self.occupancy_grid, cmap='binary')
        plt.show()
