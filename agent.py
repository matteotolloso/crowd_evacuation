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
    def __init__(self, unique_id, model, probability_optimal, alpha, beta, speed):
        super().__init__(unique_id, model)
        self.type = 'InformedPersonAgent'
        self.direction = (0, 0)
        self.probability_optimal = probability_optimal# probability of moving to the optimal position
        self.alpha = alpha
        self.beta = beta
        self.speed = speed

    def move_agent(self, new_pos):
        # compute the direction
        self.direction = utils.compute_direction(self.pos, new_pos)
        self.model.grid.move_agent(self, new_pos)
        if new_pos in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move_optimal_priority(self):

        optimal_position = self.model.static_floor_field[self.pos]

        similar_directions = utils.compute_similar_directions_3(self.pos, optimal_position)
        similar_directions.sort(key=lambda x: self.model.wall_distance[x], reverse=True)

        if self.model.random.uniform(0, 1) < self.probability_optimal: 
            # i will try to move to the optimal first
            candidates = [optimal_position] + similar_directions
        else:
            # i will try to move to similar first
            candidates = similar_directions + [optimal_position]

        for candidate in candidates:
            if (self.model.grid.is_cell_empty(candidate)):  # if it is empty, then I go
                self.move_agent(candidate)
                return True

        return False
    
    def move_with_weights(self):
        
        if self.model.random.uniform(0, 1) > self.speed:
            return True
        
        alpha = self.alpha
        beta = self.beta
        
        next_pos = []
        
        optimal = self.model.static_floor_field[self.pos]
        similar3 = utils.compute_similar_directions_3(self.pos, optimal)
        self.model.random.shuffle(similar3)
        similar5 = utils.compute_similar_directions_5(self.pos, optimal)
        self.model.random.shuffle(similar5)

        next_pos.append((optimal, beta * self.model.wall_distance[optimal] + alpha * 3 ))
        for pos in similar3:
            next_pos.append((pos, beta * self.model.wall_distance[pos] + alpha * 2))
        for pos in similar5:
            next_pos.append((pos, beta * self.model.wall_distance[pos] + alpha * 1))

        next_pos.sort(key=lambda x: x[1], reverse=True)

        for pos in next_pos:
            if (self.model.grid.is_cell_empty(pos[0])):  # if it is empty, then I go
                self.move_agent(pos[0])
                return True
        
        return False

 
    def step(self):

        is_moved = self.move_with_weights()

        if not is_moved:
            pass
            # TODO do nothing for now

        return
    

class UninformedPersonAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'UninformedPersonAgent'
        self.direction = (
            self.model.random.choice([-1, 0, 1]), 
            self.model.random.choice([-1, 0, 1]), 
        )

    def move_agent(self, new_pos):
        # compute the direction
        self.direction = utils.compute_direction(self.pos, new_pos)
        self.model.grid.move_agent(self, new_pos)
        if new_pos in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
    
    def step(self):

        # get the other agents around me
        neighbors = self.model.grid.get_neighbors(
            self.pos, 
            moore=True, 
            include_center=False, 
            radius= 1,
        )
    
        # remove the walls
        neighbors_agents = list(filter(lambda x: x.type == 'InformedPersonAgent' or x.type == 'UninformedPersonAgent', neighbors))

        self.model.random.shuffle(neighbors_agents)

        # if nobody around, move following the old direction or a random one
        if (len(neighbors_agents) == 0): 
            new_pos = (self.pos[0] + self.direction[0], self.pos[1] + self.direction[1])
            if (self.model.grid.is_cell_empty(new_pos)):  # if it is empty, then I go
                self.move_agent(new_pos)
                return

            neighbors_cells = list(self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False))
            self.model.random.shuffle(neighbors_cells)
            for cell in neighbors_cells:
                if self.model.grid.is_cell_empty(cell):
                    self.move_agent(cell)
                    return
            raise Exception('Strange situation: all the neighbors are walls ')
        
        # if someone around, move in its same direction

        direction = neighbors_agents[0].direction
        optimal_position = (self.pos[0] + direction[0], self.pos[1] + direction[1])

        
        if (self.model.grid.is_cell_empty(optimal_position)):  # if it is empty, then I go
            self.move_agent(optimal_position)
            return

        # i'm here because the optimal cell is occupied, i will try similar directions 
        similar_new_positions = utils.compute_similar_directions_3(self.pos, optimal_position)

        # random choice between the two similar cells
        self.model.random.shuffle(similar_new_positions)

        if (self.model.grid.is_cell_empty(similar_new_positions[0])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[0])
            return
        
        if (self.model.grid.is_cell_empty(similar_new_positions[1])):  # if it is empty, then I go
            self.move_agent(similar_new_positions[1])
            return


        # this means that all the meaningful directions are occupied

        # TODO do nothing for now