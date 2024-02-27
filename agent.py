import mesa
import utils 
from utils import *    


class StaticAgent(mesa.Agent):
    """
    Agent representing static elements in the building.

    Attributes:
        unique_id (int): A unique identifier for the agent.
        model (BuildingModel): Reference to the building evacuation model.
        type (str): Type of the static agent, default is 'wall'.

    Methods:
        __init__(self, unique_id, model, type='wall'):
            Initializes a StaticAgent with the specified parameters.

        step(self):
            Placeholder method for the agent's step. No action is performed.

    Notes:
        - StaticAgent instances represent static elements such as walls or exits in the building.
        - The 'type' attribute indicates the nature of the static agent (default is 'wall').
        - Static agents do not perform any actions during each model step.
    """

    def __init__(self, unique_id, model, type='wall'):
        """
        Initialize the StaticAgent.

        Parameters:
            unique_id (int): A unique identifier for the agent.
            model (BuildingModel): Reference to the building evacuation model.
            type (str): Type of the static agent, default is 'wall'.
        """
        super().__init__(unique_id, model)
        self.type = type

    def step(self):
        """
        Placeholder method for the agent's step.

        This method is called during each model step but does not perform any actions.
        """
        pass

class PersonAgent(mesa.Agent):
    """
    Agent representing a person in the building.

    Attributes:
        unique_id (int): A unique identifier for the agent.
        model (BuildingModel): Reference to the building evacuation model.
        type (str): Type of the agent, set to 'PersonAgent'.
        direction (tuple): Current movement direction of the agent.
        is_active (bool): Flag indicating whether the agent is active.

    Methods:
        __init__(self, unique_id, model):
            Initializes a PersonAgent with the specified parameters.

        move_agent(self, new_pos):
            Moves the agent to the specified position and updates its direction.
            If the new position is an exit, the agent is removed from the grid and schedule.

    Notes:
        - PersonAgent instances represent individuals within the building during an evacuation.
        - The 'type' attribute is set to 'PersonAgent'.
        - The 'direction' attribute represents the current movement direction of the agent.
        - The 'is_active' attribute indicates whether the agent is still active within the model.
    """

    def __init__(self, unique_id, model):
        """
        Initialize the PersonAgent.

        Parameters:
            unique_id (int): A unique identifier for the agent.
            model (BuildingModel): Reference to the building evacuation model.
        """
        super().__init__(unique_id, model)
        self.type = 'PersonAgent'
        self.direction = (0, 0)
        self.is_active = True
    
    def move_agent(self, new_pos):
        """
        Move the agent to the specified position and update its direction.

        If the new position is an exit, the agent is removed from the grid and schedule.

        Parameters:
            new_pos (tuple): The new position to move the agent to.
        """
        # compute the direction
        self.direction = utils.compute_direction(self.pos, new_pos)
        self.model.grid.move_agent(self, new_pos)
        if new_pos in self.model.exits: # i'm the one on the exit, just disappear
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            self.is_active = False
            self.model.active_agents -= 1


class InformedPersonAgent(PersonAgent):

    """
    Agent representing an informed person in the building with a specific evacuation strategy.

    Attributes:
        unique_id (int): A unique identifier for the agent.
        model (BuildingModel): Reference to the building evacuation model.
        type (str): Type of the agent, set to 'InformedPersonAgent'.
        direction (tuple): Current movement direction of the agent.
        alpha (float): Parameter influencing the movement decision.
        beta (float): Parameter influencing the movement decision.
        speed (float): Speed of the agent during evacuation.

    Methods:
        __init__(self, unique_id, model, alpha, beta, speed):
            Initializes an InformedPersonAgent with the specified parameters.

        move_with_weights(self):
            Calculates the next position based on a weighted decision strategy.
            Moves to the selected position if it is empty.

        step(self):
            Performs one time step for the agent.
            Calls the move_with_weights method to determine the next move.
    """

    def __init__(self, unique_id, model, alpha, beta, speed):
        """
        Initialize the InformedPersonAgent.

        Parameters:
            unique_id (int): A unique identifier for the agent.
            model (BuildingModel): Reference to the building evacuation model.
            alpha (float): Parameter influencing the movement decision.
            beta (float): Parameter influencing the movement decision.
            speed (float): Speed of the agent during evacuation.
        """
        super().__init__(unique_id, model)
        self.type = 'InformedPersonAgent'
        self.direction = (0, 0)
        self.alpha = alpha
        self.beta = beta
        self.speed = speed

    
    def move_with_weights(self):
        """
        Calculate the next position based on a weighted decision strategy.
        Move to the selected position if it is empty.

        Returns:
            bool: True if the agent successfully moves to a new position, False otherwise.
        """
        
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
        """
        Perform one time step for the agent.

        Calls the move_with_weights method to determine the next move.
        """

        if not self.is_active:
            return

        is_moved = self.move_with_weights()

        if not is_moved:
            pass

        return
    

class UninformedPersonAgent(PersonAgent):
    """
    Agent representing an uninformed person in the building with a simple evacuation strategy.

    Attributes:
        unique_id (int): A unique identifier for the agent.
        model (BuildingModel): Reference to the building evacuation model.
        type (str): Type of the agent, set to 'UninformedPersonAgent'.
        direction (tuple): Current movement direction of the agent, initialized randomly.

    Methods:
        __init__(self, unique_id, model):
            Initializes an UninformedPersonAgent with the specified parameters.

        step(self):
            Performs one time step for the agent.
            Implements a simple evacuation strategy based on the presence of nearby agents.

    Notes:
        - The 'direction' attribute is randomly initialized.
        - The agent either moves in the current direction, follows a neighbor's direction, or chooses a random direction.
    """

    def __init__(self, unique_id, model):
        """
        Initialize the UninformedPersonAgent.

        Parameters:
            unique_id (int): A unique identifier for the agent.
            model (BuildingModel): Reference to the building evacuation model.
        """

        super().__init__(unique_id, model)
        self.type = 'UninformedPersonAgent'
        self.direction = (
            self.model.random.choice([-1, 0, 1]), 
            self.model.random.choice([-1, 0, 1]), 
        )
    
    def step(self):
        """
        Perform one time step for the agent.

        Implements a simple evacuation strategy based on the presence of nearby agents.
        The agent either moves in the current direction, follows a neighbor's direction,
        or chooses a random direction if there are no neighbors.
        """

        if not self.is_active:
            return

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
