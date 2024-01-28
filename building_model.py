import mesa
from mesa.space import PropertyLayer
import numpy as np
import matplotlib.pyplot as plt
import utils 
from utils import *    
import os
import pickle
from pathlib import Path


class StaticAgent(mesa.Agent):

    def __init__(self, unique_id, model, type='wall'):
        super().__init__(unique_id, model)
        self.type = type

    def step(self):
        pass


class PersonAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'person'

    def step(self):


        # computed by A*
        optimal_position = self.model.static_floor_field[self.pos]

        if optimal_position in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

        if (self.model.grid.is_cell_empty(optimal_position)):  # if it is empty, then I go
            self.model.grid.move_agent(self, optimal_position)
            return

        # i'm here because the optimal cell is occupied, i will try similar directions 

        similar_new_positions = self.compute_similar_directions(self.pos, optimal_position)

        # random choice between the two simila cells
        self.model.random.shuffle(similar_new_positions)

        if (self.model.grid.is_cell_empty(similar_new_positions[0])):  # if it is empty, then I go
            self.model.grid.move_agent(self, similar_new_positions[0])
            return
        
        if (self.model.grid.is_cell_empty(similar_new_positions[1])):  # if it is empty, then I go
            self.model.grid.move_agent(self, similar_new_positions[1])
            return

        # this means that all the meaningful directions are occupied

        # TODO do nothing for now

        return

        
    def compute_similar_directions(self, pos1, pos2): 
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
            


class BuildingModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, path):
        super().__init__()
        self.num_agents = N
        self.path = path
        self.planimetry = utils.read_planimetry(path)
        self.width = self.planimetry.shape[1] # for numpy the width is the number of columns
        self.height = self.planimetry.shape[0]

        self.grid = mesa.space.SingleGrid(
            width=self.width, 
            height=self.height, 
            torus=False
        )

        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        self.static_agent_count = 0
        self.exits = []

        # place the walls and the exits as static agents for the sake of visualization and grid occupancy
        for i in range(self.width):
            for j in range(self.height):
                if m2n_v(self.planimetry, (i, j)) == '#':
                    a = StaticAgent(self.static_agent_count, self, type='wall')
                    self.grid.place_agent(a, (i, j))
                    self.static_agent_count += 1
                elif m2n_v(self.planimetry, (i, j)) == 'e':
                    a = StaticAgent(self.static_agent_count, self, type='exit')
                    self.grid.place_agent(a, (i, j))
                    self.static_agent_count += 1
                    self.exits.append((i, j))
                else:
                    pass
        
        # place the person agents
        for i in range(self.num_agents):
            pos = self.random.choice(list(self.grid.empties))
            a = PersonAgent(i + self.static_agent_count, self)
            self.schedule.add(a)
            self.grid.place_agent(a, pos)


        # data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Test1": lambda x:0}, agent_reporters={"Test2": lambda x:0}
        )

        self.static_floor_field = {}

        static_floor_field_path = f'./cache/static_floor_field_{Path(self.path).stem}.pkl'

        if os.path.exists(static_floor_field_path):
            self.static_floor_field = pickle.load(open(static_floor_field_path, 'rb'))
        else:
            astar = utils.AStar(self.planimetry)
            for j in range(self.height):  
                for i in range(self.width):
                    if m2n_v(self.planimetry, (i, j)) == '.' and  (i, j) not in self.static_floor_field.keys():
                        _, path = astar(
                            start = (i, j), 
                            end = self.exits[0]
                        )

                        print(f'path from {(i, j)} to {self.exits[0]}')
    
                        for iter in range(len(path) -1):
                            self.static_floor_field[path[iter]] = path[iter + 1]

            pickle.dump(self.static_floor_field, open(static_floor_field_path, 'wb'))
                        

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False