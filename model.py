import mesa
import utils 
from utils import *    
import os
import pickle
from pathlib import Path
from agent import StaticAgent, InformedPersonAgent, UninformedPersonAgent
import numpy as np


class BuildingModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, perc_uninformed_agents, alpha, beta, speed_mean, speed_variance, path = './dataset/charleston_road.txt'):
        super().__init__()
        self.num_agents = N
        self.perc_uninformed_agents = perc_uninformed_agents
        self.path = path
        self.planimetry = utils.read_planimetry(path)
        self.width = self.planimetry.shape[1] # for numpy the width is the number of columns
        self.height = self.planimetry.shape[0]
        self.grid = mesa.space.SingleGrid(
            width=self.width, 
            height=self.height, 
            torus=False
        )
        self.alpha = alpha
        self.beta = beta
        self.speed_mean = speed_mean
        self.speed_variance = speed_variance
        self.active_agents = 0

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
                    exit_agent = StaticAgent(self.static_agent_count, self, type='exit')
                    self.grid.place_agent(exit_agent, (i, j))
                    self.exits.append((i, j))
                else:
                    pass
        
        # place the person agents
                
        # uninformed agents
        number_of_uninformed_agents = int(self.perc_uninformed_agents * self.num_agents)
        for i in range(number_of_uninformed_agents):
            pos = self.random.choice(list(self.grid.empties))
            a = UninformedPersonAgent(i + self.static_agent_count, self)
            self.schedule.add(a)
            self.grid.place_agent(a, pos)
            self.active_agents += 1

        # informed agents
        number_of_informed_agents = int((1 - self.perc_uninformed_agents) * self.num_agents)
        for i in range(number_of_informed_agents):
            pos = self.random.choice(list(self.grid.empties))
            a = InformedPersonAgent(
                i + self.static_agent_count, 
                self,  
                alpha, 
                beta, 
                speed = max(0.1, np.random.normal(speed_mean, speed_variance)),
            )
            self.schedule.add(a)
            self.grid.place_agent(a, pos)
            self.active_agents += 1

        self.grid.remove_agent(exit_agent)

        static_floor_field_path = f'./cache/static_floor_field_{Path(self.path).stem}.pkl'
        if os.path.exists(static_floor_field_path):
            self.static_floor_field = pickle.load(open(static_floor_field_path, 'rb'))
        else:
            self.static_floor_field = utils.static_floor_field(self.planimetry, self.exits[0])
            pickle.dump(self.static_floor_field, open(static_floor_field_path, 'wb'))
        
        self.wall_distance = utils.distance_to_nearest_hash(self.planimetry)

        
        # data collector
        self.datacollector = mesa.DataCollector(
            model_reporters=
            {
                "Active_agents": "active_agents"
            }, 
        )

        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        # TODO stop the simulation
        # if self.active_agents == 0:
        #     self.running = False