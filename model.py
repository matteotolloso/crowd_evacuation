import mesa
import utils 
from utils import *    
import os
import pickle
from pathlib import Path
from agent import StaticAgent, InformedPersonAgent, UninformedPersonAgent
import numpy as np


class BuildingModel(mesa.Model):
    """
    Evacuation agent-based model.

    Attributes:
        num_agents (int): Total number of agents in the model.
        perc_uninformed_agents (float): Percentage of agents that are uninformed.
        path (str): Path to the planimetry file of the building.
        planimetry (numpy.ndarray): Numpy array representing the planimetry of the building.
        width (int): Width of the planimetry (number of columns).
        height (int): Height of the planimetry (number of rows).
        grid (mesa.space.SingleGrid): 2D grid representing the building layout.
        alpha (float): Parameter influencing the movement decision of informed agents.
        beta (float): Parameter influencing the movement decision of informed agents.
        speed_mean (float): Mean speed of agents during evacuation.
        speed_variance (float): Variance in speed among agents during evacuation.
        active_agents (int): Number of currently active agents in the model.
        schedule (mesa.time.RandomActivation): Scheduler for agent activation.
        running (bool): Flag indicating whether the model is running.
        static_agent_count (int): Counter for static agents (walls and exits) in the model.
        exits (list): List of exit locations in the building.
        static_floor_field (numpy.ndarray): Floor field for pathfinding.
        wall_distance (numpy.ndarray): Distance from each grid cell to the nearest wall.
        datacollector (mesa.DataCollector): Data collector for model-level metrics.

    Methods:
        __init__(self, N, perc_uninformed_agents, alpha, beta, speed_mean, speed_variance, path='./dataset/charleston_road.txt'):
            Initializes the BuildingModel with the specified parameters.

        step(self):
            Advances the model by one time step, collecting data and executing agent actions.

    """

    def __init__(self, N, perc_uninformed_agents, alpha, beta, speed_mean, speed_variance, path = './dataset/charleston_road.txt'):
        """
        Initialize the BuildingModel.

        Parameters:
            N (int): Total number of agents (both informed and uninformed).
            perc_uninformed_agents (float): Percentage of agents that are uninformed (0 to 1).
            alpha (float): Parameter influencing the movement decision of informed agents.
            beta (float): Parameter influencing the movement decision of informed agents.
            speed_mean (float): Mean speed of agents during evacuation.
            speed_variance (float): Variance in speed among agents during evacuation.
            path (str): Path to the planimetry file of the building (default: './dataset/charleston_road.txt').
        """
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
                "Active_agents": "active_agents",
                "Agents_proportion" : utils.compute_proportions
            }, 
        )

        
    def step(self):
        """
        Advance the model by one time step.

        This method performs the following actions:
        1. Collects data using the data collector.
        2. Advances the scheduler by one step, activating and updating each agent.

        The data collected includes the number of active agents and agent proportions.
        """
        self.datacollector.collect(self)
        self.schedule.step()
