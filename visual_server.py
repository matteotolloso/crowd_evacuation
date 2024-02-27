from model import BuildingModel
import mesa
from utils import value_to_html_color

width = 560
height = 410

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
    }

    if agent.type == "InformedPersonAgent":
        portrayal["Color"] = value_to_html_color(agent.speed)
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1

    if agent.type == "wall":
        portrayal["Color"] = "black"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
    
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
    {  
        "N": 3000, 
        "perc_uninformed_agents" : 0.2,
        "alpha" : 0.8,
        "beta" : 1,
        "speed_mean" : 0.7,
        "speed_variance" : 0.2,
        "path" : "./dataset/opera_teather.txt",
    }
)

server.port = 8521
server.launch(open_browser=False)