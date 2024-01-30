from model import BuildingModel
import seaborn as sns
import numpy as np
import mesa
import pandas as pd

width = 132
height = 188
agents = 2000

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Color": "orange",
        "Filled": "true",
        "Layer": 0,
        "r": 1,
    }
    if agent.type == "wall":
        portrayal["Color"] = "black"
        # portrayal["Layer"] = 1
        portrayal["r"] = 1
    
    if agent.type == "UninformedPersonAgent":
        portrayal["Color"] = "red"
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
     "perc_uninformed_agents" : 0.5,
    }
)
server.port = 8521  # the default
server.launch(open_browser=False)