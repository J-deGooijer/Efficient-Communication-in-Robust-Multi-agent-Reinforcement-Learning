import numpy as np

class HyperParameters:
    def __init__(self):
        self.N_games = 1000
        self.N_games_test = 1000
        self.print_interval = 250
        self.max_steps = 50
        self.eps = 0.1
        self.gamma_batch_size = 32
        self.gamma = 0.99
        self.tau = 0.01

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state