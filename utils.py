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
    


def compute_similar_directions(pos1, pos2): 
    """Given a start position and an end position, return similar alternative directions """
    
    if(pos1[0] == pos2[0]): # i was moving on the y axis, similar cells are the ones on the left and on the right
        return [(pos1[0] - 1, pos1[1]), (pos1[0] + 1, pos1[1])]
    if(pos1[1] == pos2[1]): # i was moving on the x axis, similar cells are the ones on the top and on the bottom
        return [(pos1[0], pos1[1] - 1), (pos1[0], pos1[1] + 1)]
    if (pos1[0] < pos2[0] and pos1[1] < pos2[1]): # i was moving on the top right corner, similar cells are the ones on the top and on the right
        return [(pos1[0], pos1[1] + 1), (pos1[0] + 1, pos1[1])]
    if (pos1[0] < pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom right corner, similar cells are the ones on the bottom and on the right
        return [(pos1[0], pos1[1] - 1), (pos1[0] + 1, pos1[1])]
    if (pos1[0] > pos2[0] and pos1[1] < pos2[1]): # i was moving on the top left corner, similar cells are the ones on the top and on the left
        return [(pos1[0], pos1[1] + 1), (pos1[0] - 1, pos1[1])]
    if (pos1[0] > pos2[0] and pos1[1] > pos2[1]): # i was moving on the bottom left corner, similar cells are the ones on the bottom and on the left
        return [(pos1[0], pos1[1] - 1), (pos1[0] - 1, pos1[1])]
    
    raise Exception('Impossible to compute similar directions, this should not be possible')

def compute_direction(pos1, pos2):
    """Given a start position and an end position, return the direction """
    
    return (pos2[0] - pos1[0], pos2[1] - pos1[1])

