from MADDPG.train_agent import Train
import time


"""
Parameters MADDPG agents:
"""
MODE = 'test' #pick "train" or "test"
LOAD = True
SAVE = False
PRINT_INTERVAL = 500
# N_GAMES = 10000
# MAX_STEPS = 50

train_agents = Train(MODE, LOAD, SAVE)

try:
    train_agents.loop()
except KeyboardInterrupt:
    train_agents.ask_save()

                











                
"""
Parameters robustness surrogates
"""
TRAIN_SVR = False



"""
Parameters lexicographic robust reinforcement learning
"""

LRRL = False


