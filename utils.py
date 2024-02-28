import numpy as np
import queue
import math
from collections import deque


def m2n_v(np_matrix, m_pos):
    '''Convert Mesa coordinates (the the bottom left cell il (0, 0) to Numpy cordinates (the bottom letf cell is (np_matrix.shape[0] - 1, 0) and get the value)'''
    x, y = m_pos
    return np_matrix[np_matrix.shape[0]-1-y, x]

def m2n_c(np_matrix, m_pos):
    '''Convert Mesa coordinates (the the bottom left cell il (0, 0) to Numpy cordinates (the bottom letf cell is (np_matrix.shape[0] - 1, 0) )'''
    x, y = m_pos
    return (np_matrix.shape[0]-1-y, x)

def n2m_c(np_matrix, n_pos):
    '''Convert Numpy coordinates (the the bottom left cell il (0, 0) to Mesa cordinates (the bottom letf cell is (np_matrix.shape[0] - 1, 0) )'''
    x, y = n_pos
    return (np_matrix.shape[0]-1-y, x)


def read_planimetry(path):
    """
    Read planimetry from a text file.

    Parameters:
        path (str): Path to the planimetry text file.

    Returns:
        numpy.ndarray: Numpy array representing the planimetry matrix.

    Reads the planimetry from a text file and converts it into a numpy array.
    Each character in the text file represents a cell in the planimetry matrix.

    Example:
    If the text file contains the following:
    ```
    ####
    #  e
    ####
    ```
    The resulting numpy array would be:
    ```
    [['#' '#' '#' '#']
     ['#' ' ' ' ' 'e']
     ['#' '#' '#' '#']]
    ```
    """
    
    with open(path, "r") as f:
        lines = f.readlines()
        python_matrix = []
        for line in lines:
            python_matrix.append([str(c) for c in line][:len(line)-1])
        
        return np.array(python_matrix, dtype=str)
    

class AStar:
    """
    A* pathfinding algorithm implementation for 2D grid navigation.

    Attributes:
        h (function): Heuristic function for estimating the cost from a given point to the goal.
        planimetry (numpy.ndarray): 2D grid representing the environment planimetry.

    Methods:
        __init__(self, planimetry, name: str = "AStar"):
            Initializes the AStar instance with the specified planimetry and heuristic function.

        __call__(self, start, end):
            Finds a path from the start point to the end point using the A* algorithm.

        get_valid_moves(self, current):
            Returns a list of valid neighboring positions for a given position on the grid.

        build_path(parent, target):
            Builds the path from the start to the target using the parent dictionary.

    Notes:
        - The planimetry should be a 2D grid represented as a numpy array where '#' represents walls.
        - The heuristic function 'h' estimates the cost from a given point to the goal.
        - The A* algorithm explores possible paths based on the provided planimetry and heuristic.
    """
    def __init__(self, planimetry, name: str = "AStar"):
        """
        Initialize the AStar instance.

        Parameters:
            planimetry (numpy.ndarray): 2D grid representing the environment planimetry.
            name (str): Name of the AStar instance (default: "AStar").
        """
        self.h = lambda start, end: math.dist(start, end)
        self.planimetry = planimetry

    def __call__(self, start, end):
        
        """
        Finds a path from the start point to the end point using the A* algorithm.

        Parameters:
            start (tuple): Coordinates of the starting point.
            end (tuple): Coordinates of the target point.

        Returns:
            tuple: A tuple containing a boolean indicating success and a list representing the path.

        If a path is found, the boolean is True, and the path is returned.
        If no path is found, the boolean is False, and the path is None.
        """

        # initialize open and close list
        open_list = queue.PriorityQueue()
        close_list = {}
        # additional dict which maintains the nodes in the open list for an easier access and check
        support_list = {}

        starting_state_g = 0
        starting_state_h = self.h(start, end)
        starting_state_f = starting_state_g + starting_state_h

        open_list.put((starting_state_f, (start, starting_state_g)))
        support_list[start] = starting_state_g
        parent = {start: None}

        while not open_list.empty():
            # get the node with lowest f
            _, (current, current_cost) = open_list.get()

            if current in close_list.keys():
                continue

            # add the node to the close list
            close_list[current] = True

            if current == end:
                path = self.build_path(parent, end)
                return True, list(path)
            
            for neighbor in self.get_valid_moves(current):
                # check if neighbor in close list, if so continue
                if neighbor in close_list.keys():
                    continue

                # compute neighbor g, h and f values
                neighbor_g = current_cost + math.dist(current, neighbor)
                neighbor_h = self.h(neighbor, end)
                neighbor_f = neighbor_g + neighbor_h
                neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
                # if neighbor in open_list
                if neighbor in support_list.keys():
                    # if neighbor_g is greater or equal to the one in the open list, continue
                    if neighbor_g >= support_list[neighbor]:
                        continue
                parent[neighbor] = current

                # add neighbor to open list and update support_list
                open_list.put(neighbor_entry)
                support_list[neighbor] = neighbor_g

        return False, None
    
    def get_valid_moves(self, current):
        """
        Returns a list of valid neighboring positions for a given position on the grid.

        Parameters:
            current (tuple): Coordinates of the current position.

        Returns:
            list: List of valid neighboring positions.
        """
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)] # Adjacent squares 
        valid_moves = []
        for move in moves:
            new_position = (current[0] + move[0], current[1] + move[1])
            if m2n_v(self.planimetry, new_position) == '#':
                continue
            valid_moves.append(new_position)
        return valid_moves

    def build_path(self, parent, target):
        """
        Builds the path from the start to the target using the parent dictionary.

        Parameters:
            parent (dict): Dictionary containing the parent-child relationships.
            target (tuple): Coordinates of the target position.

        Returns:
            list: List representing the path from start to target.
        """
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        path.reverse()
        return path
    
def static_floor_field(planimetry, exit):
    """
    Generates a static floor field for pathfinding in a given planimetry.

    Parameters:
        planimetry (numpy.ndarray): 2D grid representing the environment planimetry.
        exit (tuple): Coordinates of the exit in the planimetry.

    Returns:
        dict: Dictionary mapping positions to their next positions in the static floor field.

    Computes a static floor field for pathfinding using the A* algorithm.
    The static floor field is a dictionary where each position maps to its next position in the path.
    The generated field facilitates efficient pathfinding from any point to the exit.

    Note:
    The planimetry should be a 2D grid represented as a numpy array where '.' represents open floor space,
    '#' represents walls, and the exit is specified by its coordinates.
    """
    astar = AStar(planimetry)
    rows, cols = planimetry.shape
    
    static_floor_field = {}
    
    for i in range(cols):  
        for j in range(rows):
            if m2n_v(planimetry, (i, j)) == '.' and  (i, j) not in static_floor_field.keys():
                _, path = astar(
                    start = (i, j), 
                    end = exit
                )
                    
                for iter in range(len(path) -1):
                    static_floor_field[path[iter]] = path[iter + 1]
            
            print(f"Done {i * cols + j}/{cols * rows}")
                

    static_floor_field[exit] = exit

    return static_floor_field


def compute_similar_directions_3(pos1, pos2): 
    """Given a start position and an end position, return similar alternative directions """
    
    if(pos1[0] == pos2[0]): # i was moving on the y axis, similar cells are the ones on the left and on the right
        return [(pos2[0] - 1, pos2[1]), (pos2[0] + 1, pos2[1])]
    if(pos1[1] == pos2[1]): # i was moving on the x axis, similar cells are the ones on the top and on the bottom
        return [(pos2[0], pos2[1] - 1), (pos2[0], pos2[1] + 1)]
    
    if (pos1[0] < pos2[0] and pos1[1] < pos2[1]): # i was moving on the top right corner, similar cells are the ones on the top and on the right
        return [(pos1[0], pos1[1] + 1), (pos1[0] + 1, pos1[1])]
    if (pos1[0] < pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom right corner, similar cells are the ones on the bottom and on the right
        return [(pos1[0], pos1[1] - 1), (pos1[0] + 1, pos1[1])]
    if (pos1[0] > pos2[0] and pos1[1] < pos2[1]): # i was moving on the top left corner, similar cells are the ones on the top and on the left
        return [(pos1[0], pos1[1] + 1), (pos1[0] - 1, pos1[1])]
    if (pos1[0] > pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom left corner, similar cells are the ones on the bottom and on the left
        return [(pos1[0], pos1[1] - 1), (pos1[0] - 1, pos1[1])]
    
    raise Exception('Impossible to compute similar directions, this should not be possible')


def compute_similar_directions_5(pos1, pos2): 
    """Given a start position and an end position, return similar alternative directions """
    
    if(pos1[0] == pos2[0]): # i was moving on the y axis, 
        return [(pos1[0] - 1, pos1[1]), (pos1[0] + 1, pos1[1])]
    if(pos1[1] == pos2[1]): # i was moving on the x axis, 
        return [(pos1[0], pos1[1] - 1), (pos1[0], pos1[1] + 1)]
    
    if (pos1[0] < pos2[0] and pos1[1] < pos2[1]): # i was moving on the top right corner
        return [(pos1[0] -1, pos1[1] + 1), (pos1[0] + 1, pos1[1] - 1)]
    
    if (pos1[0] < pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom right corner
        return [(pos1[0] + 1, pos1[1] + 1), (pos1[0] - 1, pos1[1] - 1)]
    
    if (pos1[0] > pos2[0] and pos1[1] < pos2[1]): # i was moving on the top left corner
        return [(pos1[0] + 1, pos1[1] + 1), (pos1[0] - 1, pos1[1] - 1)]
    
    if (pos1[0] > pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom left corner
        return [(pos1[0] - 1, pos1[1] + 1), (pos1[0] + 1, pos1[1] -1)]
    
    raise Exception('Impossible to compute similar directions, this should not be possible')

def compute_direction(pos1, pos2):
    """Given a start position and an end position, return the direction """
    
    return [pos2[0] - pos1[0], pos2[1] - pos1[1]]


def distance_to_nearest_hash(planimetry):
    """
    Computes the distance from each open cell to the nearest '#' (hash) cell in the planimetry.

    Parameters:
        planimetry (numpy.ndarray): 2D grid representing the environment planimetry.

    Returns:
        dict: Dictionary mapping positions to their distances to the nearest '#' cell.

    Computes the distance from each open cell ('.') to the nearest '#' (hash) cell using breadth-first search.
    The function returns a dictionary where each position maps to its distance to the nearest '#' cell.

    Note:
    The planimetry should be a 2D grid represented as a numpy array where '.' represents open floor space,
    and '#' represents walls.

    """
    rows, cols = planimetry.shape
    distances = {}

    def is_valid(x, y):
        return 0 <= x < cols and 0 <= y < rows

    queue = deque()

    # Initialize distances for "#" cells and add them to the queue
    for i in range(cols):
        for j in range(rows):
            if m2n_v(planimetry, (i, j)) == "#":
                distances[(i, j)] = 0
                queue.append((i, j))
            else:
                distances[(i, j)] = -1

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and distances[(nx, ny)] == -1:
                distances[(nx, ny)] = distances[(x, y)] + math.dist((nx, ny), (x, y))
                queue.append((nx, ny))

    return distances
    

def value_to_html_color(value):
    """
    Converts a numerical value to an HTML color representation based on predefined thresholds.

    Parameters:
        value (float): Numerical value to be converted.

    Returns:
        str: HTML color representation.
    """
    if value < 0.33:
        return "red" 
    if 0.33 <= value < 0.66:
        return "orange" 
    if value >= 0.66:
        return "green" 


def compute_proportions(model):
    """
    Computes the proportions of different types of agents in the model.

    Parameters:
        model (BuildingModel): Reference to the building evacuation model.

    Returns:
        dict: Dictionary containing the count of each agent type.

    Computes and returns a dictionary containing the count of different types of agents in the model.
    The agent types include 'slow', 'medium', 'fast' for InformedPersonAgents, and 'uninformed' for UninformedPersonAgents.
    The count of 'active' agents is also included.
    """
    count = {
        "slow": 0,
        "medium" : 0,
        "fast": 0, 
        "uninformed": 0,
        "active" : model.active_agents,
    }

    for agent in model.schedule.agents:
        
        if (agent.type == 'InformedPersonAgent'):
           
            if agent.speed < 0.33:
                count["slow"] += 1
            if 0.33 <= agent.speed < 0.66:
                count["medium"] += 1
            if agent.speed >= 0.66:
                count["fast"] += 1
        else:
            
            count["uninformed"] += 1
     
    
    return count
    
    