import mesa
from model import BuildingModel
import numpy as np
import pickle

ITERATIONS = 10
MAX_STEPS = 2000

params = {    
    "N": range(100, 2100, 100), 
    "perc_uninformed_agents" : 0,
    "alpha" : 1,
    "beta" : 0.5,
    "speed_mean" : 0.8,
    "speed_variance" : 0.2,
}


result = mesa.batch_run(
    BuildingModel,
    parameters=params,
    iterations=ITERATIONS, 
    max_steps=MAX_STEPS,
    number_processes=10,
    data_collection_period=1,
    display_progress=True,
)

# dump the result 

pickle.dump(result, open("./result.pkl", "wb"))