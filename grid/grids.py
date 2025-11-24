import rasterio
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os


class Grid:
    def __init__(self, grid_file: str, cost_mapping: List) -> None:
        assert len(cost_mapping) == 14
        self.grid_file = grid_file
        self.category, self.cost = self._load_grid(grid_file, cost_mapping)
        self.categories = ["unknown", "hardened", "half-hardened", "unhardened", "track", "fallow", "agriculture",
                           "forest", "grassland", "heath", "dune", "sand", "water", "other"]
        self.colors = ["white", "black", "dimgray", "darkgray", "darkmagenta", "darkred", "orange", "forestgreen",
                       "lightgreen", "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"]
        print(f'Grid loaded with tiles: {len(self.category)}')

    def _load_grid(self, grid_file: str, cost_mapping: List) -> Tuple[np.ndarray, np.ndarray]:
        category_grid = self._load_data(grid_file)
        cost_grid = np.empty(category_grid.shape, dtype='uint8')
        for index, value in np.ndenumerate(category_grid):
            cost_grid[index] = cost_mapping[value]
        return category_grid, cost_grid

    def _load_data(self, grid_file: str) -> np.ndarray:
        with rasterio.open(grid_file) as src:
            return src.read(1).astype('uint8')

    def display_grid(self):
        color_codes = np.array([mcolors.to_rgb(c) for c in self.colors])
        color_coded = color_codes[self.category]
        file_name = os.path.basename(self.grid_file)
        plt.title(f'Index of segmented tile: {os.path.splitext(file_name)[0]}')
        plt.imshow(color_coded)
        plt.show()

