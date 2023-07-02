import numpy as np
import torch as T
from utils import obs_list_to_state_vector
from EDI.network import GammaNet



## TO DO HERE: do the Q values sheit, check the structure of sequence to make sure the slicing is done correcly in len and gamma

class DataSet:
    def __init__(self, agents):
        self.agents = agents


    def calculate_IO(self, sequence, cooperating_agents_mask):
        I = len(sequence)-1

        io = []
        for i in range(I):
            mu0 = self.get_mu(sequence[i])
            for j in range(i+1, I+1):
                zeta = sum(self.get_Q_values(sequence[j], self.get_mu(sequence[j]), cooperating_agents_mask)) - sum(self.get_Q_values(sequence[j], mu0, cooperating_agents_mask))
                for l,k in enumerate(cooperating_agents_mask):
                    gamma = np.linalg.norm(sequence[i][k]-sequence[j][k])
                    io.append([np.concatenate((sequence[i][k], sequence[j][k], np.array([zeta]))), gamma])

        # One line of IO is a set of two observations of the same agent and their corresponding Qdiff/gamma
        return io

    def get_Q_values(self, state, mu, cooperating_agents_mask):
        Q_all = []

        # Loop through agents and get Q values for state, with optimal actions for state_mu
        for i in cooperating_agents_mask: #agent_idx, agent in enumerate(self.agents.agents):
            agent = self.agents.agents[i]        
            device = self.agents.agents[i].target_critic.device
            Q = agent.target_critic.forward(T.tensor(np.array([obs_list_to_state_vector(state)]), dtype=T.float32).to(device), mu).flatten()
            Q_all.append(Q.detach().cpu().numpy()[0])
        
        # We only need the Q values for the cooperating agents
        return Q_all
    

    def get_mu(self, state_mu):
        actions = []

        # Loop through agents and get optimal actions for state_mu
        for agent_idx, agent in enumerate(self.agents.agents):
            device = self.agents.agents[agent_idx].target_actor.device
            agent_state_mu = T.tensor(np.array([state_mu[agent_idx]]), dtype=T.float32).to(device)

            action = agent.target_actor.forward(agent_state_mu)
            actions.append(action)

        mu = T.cat([acts for acts in actions], dim=1)
        return mu
