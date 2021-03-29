# -*- coding: utf-8 -*-


import copy
import math
import random
import torch as th
import numpy as np
import pandas as pd
import torch
from torch import nn
from abc import ABC
from random import choice
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation,
    Configuration,
    Action,
    row_col,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(device)


def get_nearest_food(agent_positions, food_positions, player_index):
    dist0 = pow(food_positions[0][0] - agent_positions[player_index][0][0], 2) + pow(
        food_positions[0][1] - agent_positions[player_index][0][1], 2
    )
    dist1 = pow(food_positions[1][0] - agent_positions[player_index][0][0], 2) + pow(
        food_positions[1][1] - agent_positions[player_index][0][1], 2
    )
    if dist0 < dist1:
        return food_positions[0]
    else:
        return food_positions[1]


def get_potential_moves(agent_positions):
    num_rows, num_cols = [7, 11]
    potential_moves = []
    all_potential_moves = []
    for i in range(4):
        if len(agent_positions[i]) > 0:
            north_cords = list(map(lambda x, y: x + y, (-1, 0), agent_positions[i][0]))
            north_cords = [north_cords[0] % num_rows, north_cords[1] % num_cols]
            east_cords = list(map(lambda x, y: x + y, (0, 1), agent_positions[i][0]))
            east_cords = [east_cords[0] % num_rows, east_cords[1] % num_cols]
            south_cords = list(map(lambda x, y: x + y, (1, 0), agent_positions[i][0]))
            south_cords = [south_cords[0] % num_rows, south_cords[1] % num_cols]
            west_cords = list(map(lambda x, y: x + y, (0, -1), agent_positions[i][0]))
            west_cords = [west_cords[0] % num_rows, west_cords[1] % num_cols]
            potential_moves.append(north_cords)
            potential_moves.append(east_cords)
            potential_moves.append(south_cords)
            potential_moves.append(west_cords)
            all_potential_moves.append(potential_moves)
            potential_moves = []
        else:
            all_potential_moves.append([])

    return all_potential_moves


def get_valid_moves(agent_positions, potential_moves, player_index):
    this_agent_potential_moves = potential_moves.copy()
    this_agent_potential_moves = this_agent_potential_moves[player_index]
    for i in range(4):
        if i != player_index:
            for e in this_agent_potential_moves:
                if e in potential_moves[i]:
                    this_agent_potential_moves[
                        this_agent_potential_moves.index(e)
                    ] = None

    for i in range(4):
        for e in this_agent_potential_moves:
            if e in agent_positions[i]:
                this_agent_potential_moves[this_agent_potential_moves.index(e)] = None

    return this_agent_potential_moves


def get_all_geese(observation):
    all_agent_positions = []
    intermediate_array = []
    for goose in observation.geese:
        for chunk in goose:
            cord_tuple = row_col(chunk, 11)
            intermediate_array.append(list(cord_tuple))
        all_agent_positions.append(intermediate_array)
        intermediate_array = []
    return all_agent_positions


def get_all_food(observation):
    food_positions = [-1, -1]
    for food in observation.food:
        index_in_env = (observation.food).index(food)
        cord_tuple = row_col(food, 11)
        food_positions[index_in_env] = list(cord_tuple)
    return food_positions


last_move = [None, None, None, None]




def print_table(player_index, agent_positions, food_positions, near_food, valid_moves):
    table = np.chararray((7, 11), unicode=True)
    table[:] = "9"
    for agent in agent_positions:
        for e in agent:
            table[e[0]][e[1]] = agent_positions.index(agent)


    for e in valid_moves:
        if e != None:
            table[e[0]][e[1]] = "8"

    for e in food_positions:
        table[e[0]][e[1]] = "7"

    print(table)
    
def get_table(player_index, agent_positions, food_positions, near_food, valid_moves):

    table = np.chararray((7, 11), unicode=True)
    table[:] = "9"
    for agent in agent_positions:
        for e in agent:
            table[e[0]][e[1]] = agent_positions.index(agent)


    for e in valid_moves:
        if e != None:
            table[e[0]][e[1]] = "8"

    for e in food_positions:
        table[e[0]][e[1]] = "7"

    return np.asfarray(table,float)

def agent(obs_dict, config_dict):

    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    
    player_index = observation.index

    agent_positions = get_all_geese(observation)
    food_positions = get_all_food(observation)
    near_food = get_nearest_food(agent_positions, food_positions, player_index)
    potential_moves = get_potential_moves(agent_positions)
    valid_moves = get_valid_moves(agent_positions, potential_moves, player_index)


env = make("hungry_geese")
# env.run([None, agent, agent, agent])

# print(score)
#resetting vars
last_move = [None, None, None, None]
stepsSurvived = [0, 0, 0, 0]
maxLength = [0, 0, 0, 0]
finalLength = [None, None, None, None]
score = [None, None, None, None]

#-------------------------------
#writing observation processor

class DQN(nn.Module):
    def __init__(self, n_acts):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(77, 32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(16, 4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(4, n_acts))

    def forward(self, obs):
        q_values = self.layer1(obs)
        q_values = self.layer2(q_values)
        q_values = q_values.view(-1,16)
        q_values = self.layer3(q_values)
        q_values = self.layer4(q_values)
        return q_values

    def train_on_batch(
        self,
        target_model,
        optimizer,
        obs,
        acts,
        rewards,
        next_obs,
        terminals,
        gamma=0.99,
    ):
        next_q_values = target_model.forward(next_obs)
        max_next_q_values = torch.max(next_q_values, dim=1)[0].detach()
        terminal_mods = 1 - terminals

        actual_qs = rewards + terminal_mods * gamma * max_next_q_values
        pred_qs = self.forward(obs)
        pred_qs = pred_qs.gather(index=acts.view(-1, 1), dim=1).view(-1)

        loss = torch.mean((actual_qs - pred_qs) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def add_step(self, step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def sample(self, n):
        n = min(n, len(self.data))
        indicies = np.random.choice(range(len(self.data)), n, replace=False)
        samples = np.asarray(self.data)[indicies]

        state_data = torch.tensor(np.stack(samples[:, 0])).float().to(device)
        act_data = torch.tensor(np.stack(samples[:, 1])).long().to(device)
        reward_data = torch.tensor(np.stack(samples[:, 2])).float().to(device)
        next_state_data = torch.tensor(np.stack(samples[:, 3])).float().to(device)
        terminal_data = torch.tensor(np.stack(samples[:, 4])).float().to(device)

        return state_data, act_data, reward_data, next_state_data, terminal_data
    
def get_processed_obs(obs,prev_frames):
    print(obs.observation)
    

def preprocess_obs(obs, prev_frames):
    current_obs, prev_frames = get_processed_obs(obs, prev_frames)
    return current_obs, prev_frames

def format_reward(reward):
    return reward
#begin writing training algorithm
    
trainer = env.train([None,agent,agent,agent])
obs=trainer.reset()

er_capacity = 50000
er = ExperienceReplay(er_capacity)

n_episodes = 100000;
n_acts=4
train_batch_size = 32
print_freq = 100
learning_rate = 2.5e-4
target_update_delay = 100
update_freq = 32
global_step = 1;
max_steps=200
n_anneal_steps = 1e5 # Anneal over 1m steps in paper
#epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps), 0.1, 1)
epsilon=0
model = DQN(n_acts=4).cuda()
all_rewards = []
target_model = copy.deepcopy(model)
target_model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)

obs=trainer.reset()
for episode in range(n_episodes):
    
    prev_frames=[]
    episode_reward = 0
    step=0
    done=False
    while not done:
        
        agent_positions = get_all_geese(obs)
        food_positions = get_all_food(obs)
        near_food = get_nearest_food(agent_positions, food_positions, 0)
        potential_moves = get_potential_moves(agent_positions)
        valid_moves = get_valid_moves(agent_positions, potential_moves, 0)
        player_index=0
        table = get_table(player_index, agent_positions, food_positions, near_food, valid_moves)
        table = table.reshape(77)

        if np.random.rand()<epsilon:
            act = np.random.choice(range(n_acts))
            act_num = act
        else:
            obs_tensor = torch.tensor([table]).float().to(device) 
            q_values = model(obs_tensor)[0]
            q_values = q_values.cpu().detach().numpy()
            acty = np.argmax(q_values)
            
            act_num = np.argmax(q_values)
            act = np.argmax(q_values)
        
        if act == 0: act = 'NORTH'
        if act == 1: act = 'EAST'
        if act == 2: act = 'SOUTH'
        if act == 3: act = 'WEST'
        
        next_obs, reward, done, info = trainer.step(act)
        
        episode_reward+=reward
        
        if done:
            obs = trainer.reset()
            break
        
        agent_positions = get_all_geese(next_obs)
        food_positions = get_all_food(next_obs)
        near_food = get_nearest_food(agent_positions, food_positions, 0)
        potential_moves = get_potential_moves(agent_positions)
        valid_moves = get_valid_moves(agent_positions, potential_moves, 0)
        player_index=0
        next_table = get_table(player_index, agent_positions, food_positions, near_food, valid_moves)
        next_table = next_table.reshape(77)
        
        reward=format_reward(reward)
        er.add_step([table, act_num, reward, next_table, int(done)])
        obs=next_obs #for next run
        
        if global_step % update_freq == 0:
            obs_data, act_data, reward_data, next_obs_data, terminal_data = er.sample(
                train_batch_size
            )
            model.train_on_batch(
                target_model,
                optimizer,
                obs_data,
                act_data,
                reward_data,
                next_obs_data,
                terminal_data,
            )
        if global_step and global_step % target_update_delay == 0:
            target_model = copy.deepcopy(model)
            target_model.to(device)

        step += 1

        global_step+=1
    all_rewards.append(episode_reward)
        
    if episode % print_freq == 0:
        print(
            "Episode #{} | Step #{} | Epsilon {:.2f}| Avg. Reward {:.2f}".format(
                episode,
                global_step,
                epsilon,
                np.mean(all_rewards[-print_freq:]),
            )
        )
        step += 1
    

    
    
    
    
    

