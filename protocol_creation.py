# importing necessary libraries 
import itertools
import random 
import networkx as nx    
import numpy as np 

"____________________________________________ choosing droplet path from reservoir to visit points to reservoir ____________________________________________"

"intelligent_walk allows you to compute a droplet path from point to point based on the grid layout and design chosen in config.py"

def intelligent_walk(obstacles, visit_points, reservoir_positions, gridSize, max_paths=100_000): 

    "Predefined walk for the grid based on intelligent pathfinding"

    # list to tuple conversion for networkx 
    reservoir_positions = tuple(tuple(reservoir_position) for reservoir_position in reservoir_positions)
    visit_points = tuple(tuple(visit_points) for visit_points in visit_points)
    obstacles = tuple(tuple(obstacle) for obstacle in obstacles)

    # implementing Djikstra's Algorithm using nx
    Graph = nx.grid_2d_graph(gridSize, gridSize)

    # Remove obstacles from the graph
    Graph.remove_nodes_from(obstacles)

    # Randomly select start/end reservoirs and a visit point
    start_pos, end_pos = random.sample(reservoir_positions, 2)
    visiting_point = random.sample(visit_points, 1)[0]

    try:
        # Get path from start to visit point
        all_paths_1 = list(itertools.islice(
            nx.all_shortest_paths(Graph, source=start_pos, target=visiting_point), 
            max_paths
        ))
        if not all_paths_1:
            raise nx.NetworkXNoPath(f"No path from {start_pos} to {visiting_point}")
        path_1 = random.choice(all_paths_1)
        
        # Get path from visit point to end
        all_paths_2 = list(itertools.islice(
            nx.all_shortest_paths(Graph, source=visiting_point, target=end_pos), 
            max_paths
        ))
        if not all_paths_2:
            raise nx.NetworkXNoPath(f"No path from {visiting_point} to {end_pos}")
        path_2 = random.choice(all_paths_2)
        
        # Combine paths 
        random_path = path_1 + path_2[1:]
        
    except nx.NetworkXNoPath as e:
        print(f"Warning: {e}")
        print("Falling back to direct path between reservoirs")
        # Fallback: direct path between reservoirs
        try:
            random_path = nx.shortest_path(Graph, source=start_pos, target=end_pos)
        except nx.NetworkXNoPath:
            print("Error: No valid path exists in the grid!")
            # Last resort: return just the start and end positions
            random_path = [start_pos, end_pos]

    path_coords = np.array(random_path)
    length = len(path_coords)

    # choose to uncomment line 66 if you want to see the specific path length 
    # print(f"Generated path: {start_pos} -> {visiting_point} -> {end_pos} (length: {length})")
    
    return path_coords