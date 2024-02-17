from model import BuildingModel
import seaborn as sns
import numpy as np
import mesa
import pandas as pd
from utils import value_to_html_color

width = 132
height = 188
agents = 2000

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
    }
    # change the color based on the agent.speed

    if agent.type == "InformedPersonAgent":
        portrayal["Color"] = value_to_html_color(agent.speed)
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1

    if agent.type == "wall":
        portrayal["Color"] = "black"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1

        # return None # TODO remove
    
    if agent.type == "UninformedPersonAgent":
        portrayal["Color"] = "blue"
        portrayal["r"] = 1
        # portrayal["Layer"] = 1

    if agent.type == "path":
        portrayal["Color"] = "blue"
        portrayal["r"] = 1
        # portrayal["Layer"] = 1

    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, width, height, 1000, 1000)
 
server = mesa.visualization.ModularServer(
    BuildingModel, [grid], "Building_model", 
    {"N": agents, 
     "path" : './dataset/charleston_road.txt',
     "perc_uninformed_agents" : 0.05,
     "probability_optimal" : 0.2,
     "alpha" : 1,
     "beta" : 0.5,
     "speed_mean" : 0.7,
     "speed_variance" : 0.2,
    }
)
server.port = 8521  # the default
server.launch(open_browser=False)