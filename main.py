from grid import Grid, OccupancyGrid, cell_to_rd, rd_to_gps
from path_planners.grid_based import AStar
import json

if __name__ == '__main__':
    tile_index = "004"
    input_data = f"C:/MTP-Data/test_enschede/tiles/predictions/{tile_index}.tif"
    reduction_factor = 1
    spacing = 15        # given in meters
    start_location = (int(2080/reduction_factor), int(544/reduction_factor))
    goal_location = (int(5968/reduction_factor), int(5152/reduction_factor))
    # The minimum cost is set to 2 for the hardened road. The rest follows from this
    cost_list = [100,   # unknown
                 1,     # hardened
                 1.2,   # half-hardened
                 1.5,   # unhardened
                 2,     # track
                 5,     # fallow
                 3,     # agriculture
                 4,     # forest
                 2.5,     # grassland
                 2.5,     # heath
                 3.5,   # dune
                 5,     # sand
                 100,   # water
                 10     # other
                 ]

    grid = Grid(input_data, cost_list)
    # grid.display_grid()
    a_star = AStar(grid)
    # a_star.run_interactive()
    a_star.start_search(start_location, goal_location)
    a_star.display_path_on_grid(export=True, title=f"Tile {tile_index}, reduction factor {reduction_factor}",
                                save_path=f"tile_{tile_index}_{reduction_factor}.jpg")
    if len(a_star.path) > 0:
        rd_path = cell_to_rd(a_star.path, grid.resolution, grid.bounds, int(spacing/(0.1*reduction_factor)))
        gps_coords = rd_to_gps(rd_path)
        json.dump(gps_coords, open(f'gps_coords_{tile_index}_{reduction_factor}.json', 'w'))

    print("Done")
