from grid import Grid
from path_planners import AStar

if __name__ == '__main__':
    input_data = "C:/MTP-Data/gpp_dataset/soesterberg/predictions/001.tif"
    cost_dict = {
        "unknown": 0,
        "hardened": 0.1,
        "half-hardened": 0.2,
        "unhardened": 0.5,
        "track": 1,
        "fallow": 2,
        "agriculture": 3,
        "forest": 5,
        "grassland": 2,
        "heath": 2,
        "dune": 4,
        "sand": 4,
        "water": 255,
        "other": 255
    }

    cost_list = [cost for category, cost in cost_dict.items()]
    grid = Grid(input_data, cost_list)
    grid.reduce_grid((0, 0), (5000, 5000))
    grid.display_grid()

    astar = AStar(grid)
    astar.run_interactive()