import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
q_table = np.zeros((nb_states, nb_actions)) # ajanın beyni

print("Q-Table")
print(q_table)

episodes = 1000
alpha = 0.5
gamma = 0.9

outcomes = []


# training

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # ajanın başarı durumu
    outcomes.append("Failure")
    
    while not done: # ajan başarılı olana kadar state içerisinde hareket et(action sec ve uygula)
    
        # action
        if np.max(q_table[state]) > 0:
            action = np.argmax(q_table[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
        
        # update q table
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table(state, action))
            
        state = new_state
        
        if reward:
            outcomes[-1] =  "Success"
            
        
print("QTable After Training: ")
print(q_table)

plt.bar(range(episodes), outcomes)








































