# from maddpg import MADDPG
# from make_env import make_env
# import time
# import torch as T
# import numpy as np

# def obs_list_to_state_vector(observation):
#     state = np.array([])
#     for obs in observation:
#         state = np.concatenate([state, obs])
#     return state



# scenario = 'simple_adversary'
# env = make_env(scenario)
# n_agents = env.n
# actor_dims = []
# for i in range(n_agents):
#     actor_dims.append(env.observation_space[i].shape[0])
# critic_dims = sum(actor_dims)


# # action space is a list of arrays, assume each agent has same action space
# n_actions = env.action_space[0].n
# maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
#                         fc1=64, fc2=64,  
#                         alpha=0.01, beta=0.01, scenario=scenario,
#                         chkpt_dir='tmp/maddpg/')


# maddpg_agents.load_checkpoint()
# obs = env.reset()

# for i in range(4):
#     env.render()

#     # print("observations", obs)
#     actions = maddpg_agents.eval_choose_action(obs)
#     # print("actions:", actions)
#     # print("")

#     # device = maddpg_agents.agents[0].actor.device
#     # agent = maddpg_agents.agents[0]

#     # print(obs)
#     # print(obs_list_to_state_vector(obs))

#     # states_ = T.tensor(obs_list_to_state_vector(obs), dtype=T.float).to(device)
#     # actions_ = T.tensor(actions, dtype=T.float).to(device)
    
#     # agents_actions = []
#     # for agent_idx, agent in enumerate(maddpg_agents.agents):
#     #     agents_actions.append(actions_[agent_idx])
#     # print(agents_actions)
#     # actions__ = T.cat([acts for acts in agents_actions], dim=1)

#     # test = agent.target_critic.forward(states_, actions__).flatten()
#     # print(test)

#     obs, reward, done, info = env.step(actions)
#     input("Press Enter to continue...")

from pettingzoo.mpe import simple_adversary_v2

env = simple_adversary_v2.env(render_mode="human", N=2, max_cycles=25, continuous_actions=True)

env.reset()

# print(env.agent_iter)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     print(agent)
#     action = env.action_space(agent).sample()
#     print(env.observation_space(agent))
#     env.step(action)
#     # env.render()

print(env.observation_space(env.possible_agents[2]).shape)

# print(env.observation_space[0])
