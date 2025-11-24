from grid import Grid

if __name__ == '__main__':
    input_data = "C:/MTP-Data/gpp_dataset/soesterberg/predictions/001.tif"
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    grid = Grid(input_data, cost_list)
    grid.display_grid()