import mesa
import utils 
from utils import *    


class StaticAgent(mesa.Agent):

    def __init__(self, unique_id, model, type='wall'):
        super().__init__(unique_id, model)
        self.type = type

    def step(self):
        pass


class InformedPersonAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'InformedPersonAgent'
        self.direction = (0, 0)

    def move_agent(self, new_pos):
        # compute the direction
        self.direction = utils.compute_direction(self.pos, new_pos)
        self.model.grid.move_agent(self, new_pos)
        if new_pos in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
 
    def step(self):

        # computed by A*
        optimal_position = self.model.static_floor_field[self.pos]


        if (self.model.grid.is_cell_empty(optimal_position)):  # if it is empty, then I go
            self.move_agent(optimal_position)
            return

        # i'm here because the optimal cell is occupied, i will try similar directions 

        similar_new_positions = utils.compute_similar_directions(self.pos, optimal_position)

        # random choice between the two simila cells
        self.model.random.shuffle(similar_new_positions)

        if (self.model.grid.is_cell_empty(similar_new_positions[0])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[0])
            return
        
        if (self.model.grid.is_cell_empty(similar_new_positions[1])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[1])
            return

        # this means that all the meaningful directions are occupied

        # TODO do nothing for now

        return
    

class UninformedPersonAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'UninformedPersonAgent'
        self.direction = (0, 0)

    def move_agent(self, new_pos):
        # compute the direction
        self.direction = utils.compute_direction(self.pos, new_pos)
        self.model.grid.move_agent(self, new_pos)
        if new_pos in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
    
    def step(self):

        # get the other people around me
        neighbors = self.model.grid.get_neighbors(
            self.pos, 
            moore=True, 
            include_center=False, 
            radius= 1,  # TODO add as a parameter
        )
    
        # remove the walls
        neighbors = list(filter(lambda x: x.type == 'InformedPersonAgent' or x.type == 'UninformedPersonAgent', neighbors))

        self.model.random.shuffle(neighbors)

        # if nobody around, move to a random position
        if (len(neighbors) == 0): 
            neighbors_positions = list(self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False))
            self.model.random.shuffle(neighbors_positions)
            for cell in neighbors_positions:
                if self.model.grid.is_cell_empty(cell):
                    self.move_agent(cell)
                    return
            print('strange')
            raise Exception('Strange situation: all the neighbors are walls ')
        
        # if someone around, move in its same direction

        direction = neighbors[0].direction
        optimal_position = (self.pos[0] + direction[0], self.pos[1] + direction[1])

        
        if (self.model.grid.is_cell_empty(optimal_position)):  # if it is empty, then I go
            self.move_agent(optimal_position)
            return

        # i'm here because the optimal cell is occupied, i will try similar directions 
        similar_new_positions = utils.compute_similar_directions(self.pos, optimal_position)

        # random choice between the two simila cells
        self.model.random.shuffle(similar_new_positions)

        if (self.model.grid.is_cell_empty(similar_new_positions[0])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[0])
            return
        
        if (self.model.grid.is_cell_empty(similar_new_positions[1])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[1])
            return


        # this means that all the meaningful directions are occupied

        # TODO do nothing for now