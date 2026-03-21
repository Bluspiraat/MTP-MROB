from grid import Grid, OccupancyGrid, cell_to_rd, rd_to_gps
from path_planners.grid_based import AStar
import json

if __name__ == '__main__':
    input_data = "C:/MTP-Data/test_enschede/tiles/predictions/001.tif"
    reduction_factor = 8
    spacing = 15        # given in meters
    start_location = (int(9200/reduction_factor), int(8000/reduction_factor))
    goal_location = (int(6500/reduction_factor), int(9500/reduction_factor))
    # The minimum cost is set t 1 for the hardened road. The rest follows from this
    cost_list = [100,   # unknown
                 1,     # hardened
                 1.2,   # half-hardened
                 1.5,   # unhardened
                 2,     # track
                 3,     # fallow
                 3,     # agriculture
                 4,     # forest
                 2.5,     # grassland
                 2.5,     # heath
                 3.5,   # dune
                 5,     # sand
                 100,   # water
                 10     # other
                 ]

    grid = Grid(input_data, cost_list).resample(reduction_factor)
    grid.display_grid()
    a_star = AStar(grid)
    a_star.start_search(start_location, goal_location)
    a_star.display_path_on_grid()
    if len(a_star.path) > 0:
        rd_path = cell_to_rd(a_star.path, grid.resolution, grid.bounds, int(spacing/(0.1*reduction_factor)))
        gps_coords = rd_to_gps(rd_path)
        json.dump(gps_coords, open('gps_coords.json', 'w'))
        print(gps_coords)

    print("Done")
