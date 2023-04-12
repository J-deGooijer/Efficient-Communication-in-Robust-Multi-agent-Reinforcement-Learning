import numpy as np
from MADDPG.maddpg import MADDPG
from MADDPG.buffer import MultiAgentReplayBuffer
from make_env import make_env
import time

class Train:
    def __init__(self, mode='train', load=False, save=True, print_interval=500, N_games=50000, max_steps=50):
        self.mode = mode
        if self.mode=='test':
            self.load = True
            self.save = False
        elif self.mode=='train':
            self.load = load
            self.save = save
        else:
            raise Exception("Wrong mode selected, please choose test or train")
    
        self.print_interval = print_interval
        self.N_games = N_games
        self.max_steps = max_steps

        scenario = 'simple_adversary'
        self.env = make_env(scenario)
        self.n_agents = self.env.n

        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        self.n_actions = self.env.action_space[0].n
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, self.n_actions, 
                            fc1=64, fc2=64,  
                            alpha=0.01, beta=0.01, scenario=scenario,
                            chkpt_dir='MADDPG/tmp/maddpg/')

        self.memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                            self.n_actions, self.n_agents, batch_size=1024)
        
        
    def loop(self):
        total_steps = 0
        score_history = []
        best_score = 0

        if self.load:
            self.maddpg_agents.load_checkpoint()

        for i in range(self.N_games):
            obs = self.env.reset()
            score = 0
            done = [False]*self.n_agents
            episode_step = 0

            while not any(done):
                if self.mode=='test':
                    self.env.render()
                    time.sleep(0.1)
                    actions = self.maddpg_agents.eval_choose_action(obs)
                else:
                    actions = self.maddpg_agents.choose_action(obs)
                obs_, reward, done, info = self.env.step(actions)

                state = self.obs_list_to_state_vector(obs)
                state_ = self.obs_list_to_state_vector(obs_)

                if episode_step >= self.max_steps:
                    done = [True]*self.n_agents

                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)

                if total_steps % 100 == 0 and self.mode=='train':
                    self.maddpg_agents.learn(self.memory)

                obs = obs_

                score += sum(reward)
                total_steps += 1
                episode_step += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-300:])

            if self.mode=='train':
                if avg_score > best_score:
                    if self.save and i>300:
                        self.maddpg_agents.save_checkpoint()
                        if np.std(score_history[-2000:]) <= 0.3:
                            print("Models trained, switching mode to testing")
                            self.mode = 'test'
                    best_score = avg_score                    
            else:
                print("Average score: ", avg_score)
            if i % self.print_interval == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))

        print("Final best score: ", best_score)
        self.ask_save()


    def ask_save(self):
        answer = False
        while not answer:
            user_input = input("Would you like to save the models? (y/n) ")
            if user_input.lower() == 'y':
                self.maddpg_agents.save_checkpoint()
                answer = True
            elif user_input.lower() == 'n':
                answer = True
                pass
            else:
                print("Invalid reply, please respond y or n")


    def force_save(self):
        self.maddpg_agents.save_checkpoint()


    @staticmethod
    def obs_list_to_state_vector(observation):
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state