import numpy as np
import queue
import math


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
    
    with open(path, "r") as f:
        lines = f.readlines()
        python_matrix = []
        for line in lines:
            python_matrix.append([str(c) for c in line][:len(line)-1])
        
        return np.array(python_matrix, dtype=str)
    

class AStar:
    def __init__(self, planimetry, name: str = "AStar"):
        self.h = lambda start, end: math.dist(start, end)
        self.planimetry = planimetry

    def __call__(self, start, end):

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
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)] # Adjacent squares 
        valid_moves = []
        for move in moves:
            new_position = (current[0] + move[0], current[1] + move[1])
            if m2n_v(self.planimetry, new_position) == '#':
                continue
            valid_moves.append(new_position)
        return valid_moves

    @staticmethod
    def build_path(parent, target):
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        path.reverse()
        return path

    


# class Node():
#     """A node class for A* Pathfinding"""

#     def __init__(self, parent=None, position=None):
#         self.parent = parent
#         self.position = position

#         self.g = 0
#         self.h = 0
#         self.f = 0

#     def __eq__(self, other):
#         return self.position == other.position


# def astar(maze, start, end):
#     """Returns a list of tuples as a path from the given start to the given end in the given maze"""

#     print('calculating the best path from', start, 'to', end)

#     # Create start and end node
#     start_node = Node(None, start)
#     start_node.g = start_node.h = start_node.f = 0
#     end_node = Node(None, end)
#     end_node.g = end_node.h = end_node.f = 0

#     # Initialize both open and closed list
#     open_list = []
#     closed_list = []

#     # Add the start node
#     open_list.append(start_node)

#     # Loop until you find the end
#     while len(open_list) > 0:

#         # Get the current node
#         current_node = open_list[0]
#         current_index = 0
#         for index, item in enumerate(open_list):
#             if item.f < current_node.f:
#                 current_node = item
#                 current_index = index

#         # Pop current off open list, add to closed list
#         open_list.pop(current_index)
#         closed_list.append(current_node)

#         # Found the goal
#         if current_node == end_node:
#             path = []
#             current = current_node
#             while current is not None:
#                 path.append(current.position)
#                 current = current.parent
#             return path[::-1] # Return reversed path

#         # Generate children
#         children = []
#         for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

#             # Get node position
#             node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

#             # Make sure within range
#             if node_position[0] > (maze.shape[0] - 1) or node_position[0] < 0 or node_position[1] > (maze.shape[1]  -1) or node_position[1] < 0:
#                 continue

#             # Make sure walkable terrain
#             if maze[node_position[0]][node_position[1]] == '#':
#                 continue

#             # Create new node
#             new_node = Node(current_node, node_position)

#             # Append
#             children.append(new_node)

#         # Loop through children
#         for child in children:

#             # Child is on the closed list
#             for closed_child in closed_list:
#                 if child == closed_child:
#                     continue

#             # Create the f, g, and h values
#             child.g = current_node.g + 1
#             child.h = 0  #np.sqrt( ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
#             child.f = child.g + child.h

#             # Child is already in the open list
#             for open_node in open_list:
#                 if child == open_node and child.g > open_node.g:
#                     continue

#             # Add the child to the open list
#             open_list.append(child)
