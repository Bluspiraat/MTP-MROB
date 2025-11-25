from grid import Grid, OccupancyGrid

if __name__ == '__main__':
    input_data = "C:/MTP-Data/gpp_dataset/soesterberg/predictions/001.tif"
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    grid = Grid(input_data, cost_list)
    occupancy_grid = OccupancyGrid(grid, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    occupancy_grid.display_grid()
