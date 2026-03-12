from grid import Grid, OccupancyGrid, cell_to_rd, rd_to_gps
from path_planners.grid_based import AStar

if __name__ == '__main__':
    input_data = "C:/MTP-Data/gpp_dataset/soesterberg/predictions/001.tif"
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    grid = Grid(input_data, cost_list)
    grid.reduce_grid((2500, 2500), (5000, 5000))
    a_star = AStar(grid)
    a_star.start_search((1000, 500), (1000, 4000))
    # a_star.display_path_on_grid()
    rd_path = cell_to_rd(a_star.path, grid.resolution, grid.bounds)
    gps_coords = rd_to_gps(rd_path)
    print("Done")
