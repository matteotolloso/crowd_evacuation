import mesa
import utils 
from utils import *    
import os
import pickle
from pathlib import Path
from agent import StaticAgent, InformedPersonAgent, UninformedPersonAgent


class BuildingModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, path, perc_uninformed_agents, probability_optimal):
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
        for i in range(int(self.perc_uninformed_agents * self.num_agents)):
            pos = self.random.choice(list(self.grid.empties))
            a = UninformedPersonAgent(i + self.static_agent_count, self)
            self.schedule.add(a)
            self.grid.place_agent(a, pos)

        # informed agents
        for i in range(int(( 1- self.perc_uninformed_agents) * self.num_agents)):
            pos = self.random.choice(list(self.grid.empties))
            a = InformedPersonAgent(i + self.static_agent_count, self, probability_optimal)
            self.schedule.add(a)
            self.grid.place_agent(a, pos)

        self.grid.remove_agent(exit_agent)

        # data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Test1": lambda x:0}, agent_reporters={"Test2": lambda x:0}
        )

        static_floor_field_path = f'./cache/static_floor_field_{Path(self.path).stem}.pkl'
        if os.path.exists(static_floor_field_path):
            self.static_floor_field = pickle.load(open(static_floor_field_path, 'rb'))
        else:
            self.static_floor_field = utils.static_floor_field(self.planimetry, self.exits)
            pickle.dump(self.static_floor_field, open(static_floor_field_path, 'wb'))
                        

        self.wall_distance = utils.distance_to_nearest_hash(self.planimetry)

        print(len(self.wall_distance.keys()))
        print(self.height * self.width )

        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False