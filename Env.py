# importing necessary libraries 
from gymnasium import Env
import numpy as np
import json
import random
import gymnasium.spaces as spaces
from collections import deque


# importing helper functions and other modules
from helper_functions import voltage_grid, get_obs
from protocol_creation import intelligent_walk
from rendering import initialize_render, render_env

# importing variables from other files 
with open('grid_configuration.json','r') as f: 
    grid_configuration = json.load(f)
globals().update(grid_configuration)

" __________________________________________ customizable environment in Gymnasium __________________________________________ "

" In this script, all RL related tasks are defined and editable."

class VoltageDistributionEnv(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": render_fps}

    def __init__(self, render_mode=None, size=gridSize): 
        super().__init__()
        self.size = size
        self.render_mode = render_mode 
        self.window_size = window_size
        self.N = N 
        self.electrode_positions = electrode_positions

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._droplet_location = np.array([0, 0], dtype=np.int32)
        self._voltage_grid = np.zeros((size, size))

        # format of the observations the agent will recieve
        self.observation_space = spaces.Dict({
            'voltage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'probe position': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),  
            'voltage history': spaces.Box(low=-1, high=1, shape=(self.N, size, size), dtype=np.float32),
            'time step': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

         # agent only moves in the set of valid positions 
        self.action_space = spaces.Discrete(len(electrode_positions))

        self._droplet_path = np.empty(((size*2)-1, 2), dtype=np.int32) # holds empty set for droplet path 
        self.agent_location_history = []  # Tracks agent location history

        self.voltage_history = deque([np.full((self.size, self.size), -1.0) for _ in range(self.N)], maxlen=self.N)  # Tracks voltage history 

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def step(self, action):
        """the "step" function advances the environment to the next state based on the current action taken by the agent"""
        "we also define the reward function here, which is used to train the agent"
        
        # update agent's location based on action
        self._agent_location = np.array(electrode_positions[action], dtype=np.int32)
        self.agent_location_history.append(self._agent_location)

        # finding voltage at the new position
        voltage_at_new_position = self._voltage_grid[self._agent_location[0], self._agent_location[1]]


        empty_grid = np.full((self.size, self.size), -1.0)
        empty_grid[self._agent_location[0], self._agent_location[1]] = voltage_at_new_position
        self.voltage_history.append(empty_grid)


        terminated = False  # initial termination flag
        voltage_reward_weight = 1


        # -------------------------------- Reward Engineering -------------------------------- # 
        reward = 0 # initial reward value 

        voltage_reward = voltage_at_new_position * voltage_reward_weight  # reward based on voltage at the agent's position
        reward += voltage_reward
    
        # ------------------------------------------------------------------------------------ # 

        observation = get_obs(self._voltage_grid, self._agent_location, self._droplet_path_index, np.stack(list(self.voltage_history), axis=0), self.length)

        # update droplet location based on intelligent random walk  
        self._droplet_path_index += 1
        if self._droplet_path_index < len(self._droplet_path):
            self._droplet_location = self._droplet_path[self._droplet_path_index]
            self._voltage_grid = voltage_grid(self._droplet_location, sigma, obstacles, gridSize)
        

        # terminate once droplet has reached ending position
        terminated = np.array_equal(self._droplet_location, self.end_pos)
        info = {}

        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """reset function returns the environment to its initial starting state"""

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.reward = 0  # Reset the reward
        self.agent_location_history = []  # Reset the agent location history
        self.voltage_history = deque([np.full((self.size, self.size), -1.0) for _ in range(self.N)], maxlen=self.N)


        # predefine the droplet path 
        self._droplet_path = intelligent_walk(obstacles, visit_points, reservoir_positions, gridSize)

        # initializing droplet path index 
        self._droplet_path_index = 0
        self._droplet_location = self._droplet_path[self._droplet_path_index]

        # store end position for termination check
        self.end_pos = self._droplet_path[-1]
        self.length = len(self._droplet_path)

        # Choosing the agent's location uniformly at random
        self._agent_location = np.array(random.choice(electrode_positions), dtype=np.int32)


        # Ensure agent and droplet are not in the same location
        while np.array_equal(self._agent_location, self._droplet_location):
            self._agent_location = np.array(random.choice(electrode_positions), dtype=np.int32)

        self.agent_location_history.append(self._agent_location) # Recording initial location

        # populating the grid with voltage values
        self._voltage_grid = voltage_grid(self._droplet_location, sigma, obstacles, gridSize)

        # record initial voltage
        voltage_at_start = self._voltage_grid[self._agent_location[0], self._agent_location[1]]
        empty_grid = np.full((self.size, self.size), -1.0)
        empty_grid[self._agent_location[0], self._agent_location[1]] = voltage_at_start
        self.voltage_history.append(empty_grid)

        # recording observations 
        observation = get_obs(self._voltage_grid, self._agent_location, self._droplet_path_index, np.stack(list(self.voltage_history), axis=0), self.length)

        info = {}  
        return observation, info

    def render(self):
        """visualizes the grid environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """rendering the environment"""
        # Initialize rendering if needed
        self.window, self.clock = initialize_render(self.render_mode, self.window, self.window_size)
        
        # Render the environment
        result = render_env(
            self._voltage_grid, 
            self._agent_location, 
            self._droplet_location, 
            obstacles, 
            self.window, 
            self.clock, 
            self.render_mode, 
            gridSize, 
            window_size,
            render_fps  # Use configurable FPS
        )
        
        return result

    def close(self):
        """Clean up rendering resources"""
        from rendering import cleanup_render
        cleanup_render()