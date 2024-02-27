import mesa
from model import BuildingModel
import numpy as np
import pickle

ITERATIONS = 10
MAX_STEPS = 10000

params = {    
    "N": range(1000, 20000, 2000 ), 
    "perc_uninformed_agents" : 0,
    "alpha" : 1,
    "beta" : 0.5,
    "speed_mean" : np.arange(0.1, 1, 0.2),
    "speed_variance" : 0.4,
}


result = mesa.batch_run(
    BuildingModel,
    parameters=params,
    iterations=ITERATIONS, 
    max_steps=MAX_STEPS,
    number_processes=12,
    data_collection_period=1,
    display_progress=True,
)

pickle.dump(result, open("./result.pkl", "wb"))