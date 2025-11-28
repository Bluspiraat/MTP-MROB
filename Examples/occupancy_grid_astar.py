from grid import Grid, OccupancyGrid
from path_planners import AStarOccupancy

if __name__ == '__main__':
    input_data = "C:/MTP-Data/gpp_dataset/soesterberg/predictions/001.tif"
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    grid = Grid(input_data, cost_list)
    grid.reduce_grid((0, 0), (5000, 5000))
    grid.display_grid()

    only_roads = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    occupancy_grid = OccupancyGrid(grid, only_roads)
    occupancy_grid.display_grid()

    astar = AStarOccupancy(occupancy_grid)
    astar.run_interactive()