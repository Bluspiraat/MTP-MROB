import rasterio
import numpy as np
from typing import Dict, Tuple, List
from numpy.typing import NDArray
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field


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

    # --- Standard values --- #
    categories = ["unknown", "hardened", "half-hardened", "unhardened", "track", "fallow", "agriculture",
                  "forest", "grassland", "heath", "dune", "sand", "water", "other"]
    colors = ["white", "black", "dimgray", "darkgray", "darkmagenta", "darkred", "orange", "forestgreen",
              "lightgreen", "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"]

    # --- Main maps --- #
    category: NDArray = field(init=False)
    cost: NDArray = field(init=False)

    def __post_init__(self):
        assert len(self.cost_mapping) == 14
        self.category, self.cost = self._load_grid(self.cost_mapping)
        print(f'Grid loaded with tiles: {len(self.category)}')

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
            return src.read(1).astype('uint8')

    def display_grid(self):
        """
        Displays the grid in a matplotlib figure.

        """
        color_codes = np.array([mcolors.to_rgb(c) for c in self.colors])
        color_coded = color_codes[self.category]
        file_name = os.path.basename(self.grid_file)
        plt.title(f'Index of segmented tile: {os.path.splitext(file_name)[0]}')
        plt.imshow(color_coded)
        plt.show()

    # def get_neighbours(self, tile: Tuple[int, int]) -> List[Tuple[int, int]]:


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

    def __post_init__(self):
        """
        Converts the existing grid in the Grid object into a binary occupancy grid.

        Returns:
            occupancy_grid (np.ndarray[bool]): A 2D array of occupancy.
        """
        lookup = np.zeros(14, dtype=bool)
        lookup[self.blocked_categories] = True
        self.occupancy_grid = lookup[self.base_grid.category]

    def display_grid(self):
        blocked_categories = [self.base_grid.categories[index] for index in self.blocked_categories]
        plt.title(f'Occupancy grid with blocked categories: {blocked_categories}')
        plt.imshow(self.occupancy_grid, cmap='binary')
        plt.show()
