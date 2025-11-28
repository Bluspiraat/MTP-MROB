from grid import Grid
from path_planners import AStar

if __name__ == '__main__':
    # --- Setup --- #
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
    start = (3872, 208)
    goal = (9092, 9040)
    cost_list = [cost for category, cost in cost_dict.items()]
    grid = Grid(input_data, cost_list)
    print("Created grid")
    grid_2 = grid.resample(2)
    grid_4 = grid.resample(4)
    grid_8 = grid.resample(8)
    print("Grids created")
    astar = AStar(grid)
    astar_2 = AStar(grid_2)
    astar_4 = AStar(grid_4)
    astar_8 = AStar(grid_8)

    # --- Compute paths --- #
    astar.start_search(start, goal)
    astar_2.start_search((int(start[0] / 2), int(start[1] / 2)), (int(goal[0] / 2), int(goal[1] / 2)))
    astar_4.start_search((int(start[0] / 4), int(start[1] / 4)), (int(goal[0] / 4), int(goal[1] / 4)))
    astar_8.start_search((int(start[0] / 8), int(start[1] / 8)), (int(goal[0] / 8), int(goal[1] / 8)))

    # -- Show paths --- #
    astar.display_path_on_grid()
    astar_2.display_path_on_grid()
    astar_4.display_path_on_grid()
    astar_8.display_path_on_grid()
