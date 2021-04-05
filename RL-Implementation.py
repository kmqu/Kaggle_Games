# -*- coding: utf-8 -*-
#%%

import copy
import math
import random
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

#%%


# helper function to determine the opposite move the input. Returns values determined by the kaggle env.


def opposite(subject):
    if subject == "NORTH":
        return Action.SOUTH.name
    if subject == "EAST":
        return Action.WEST.name
    if subject == "SOUTH":
        return Action.NORTH.name
    if subject == "WEST":
        return Action.EAST.name


# function returns the position of the closest food out of the guaranteed two.
# implements l2 norm without the square root.


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


# helper function to determine the 4 available for all 4 agents.


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


# funtion accepts output from potential moves and returns valid (non terminating) moves
# by excluding the opponents' current position


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


# returns the position of all agents as an array of tuples


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


# returns the position of all food as an array of tuples


def get_all_food(observation):
    food_positions = [-1, -1]
    for food in observation.food:
        index_in_env = (observation.food).index(food)
        cord_tuple = row_col(food, 11)
        food_positions[index_in_env] = list(cord_tuple)
    return food_positions


last_move = [None, None, None, None]


# this functions does most of the heavy lifting. It is responsible for returning the action
# to be taken by the current agent. It does so by selecting the move that will minimize
# the distance between the agent's head-position and the nearest food. The logic will
# ensure the agent does not collide with opponents unless there is no other possibility


def get_move(
    agent_positions,
    food_positions,
    potential_moves,
    valid_moves,
    near_food,
    player_index,
):
    go_passive = False

    for i in range(4):
        for food in food_positions:
            if i != player_index:
                if food in agent_positions[i]:
                    go_passive = True

    if go_passive == False:
        for food in food_positions:
            if len(potential_moves[player_index]) != 0:
                if food in potential_moves[player_index]:
                    if (
                        potential_moves[player_index].index(food) == 0
                        and last_move[player_index] != Action.SOUTH.name
                    ):
                        x = Action.NORTH.name
                        last_move[player_index] = x
                        return x
                    if (
                        potential_moves[player_index].index(food) == 1
                        and last_move[player_index] != Action.WEST.name
                    ):
                        x = Action.EAST.name
                        last_move[player_index] = x
                        return x
                    if (
                        potential_moves[player_index].index(food) == 2
                        and last_move[player_index] != Action.NORTH.name
                    ):
                        x = Action.SOUTH.name
                        last_move[player_index] = x
                        return x
                    if (
                        potential_moves[player_index].index(food) == 3
                        and last_move[player_index] != Action.EAST.name
                    ):
                        x = Action.WEST.name
                        last_move[player_index] = x
                        return x

    if valid_moves[0] != None:
        valid_moves[0] = Action.NORTH.name
    if valid_moves[1] != None:
        valid_moves[1] = Action.EAST.name
    if valid_moves[2] != None:
        valid_moves[2] = Action.SOUTH.name
    if valid_moves[3] != None:
        valid_moves[3] = Action.WEST.name

    valid_moves = [x for x in valid_moves if x != None]
    valid_moves = [x for x in valid_moves if x != opposite(last_move[player_index])]

    num_rows, num_cols = [7, 11]
    food_row, food_col = near_food
    player_row, player_col = agent_positions[player_index][0]

    north_dist = (-food_row + player_row) % num_rows
    east_dist = (food_col - player_col) % num_cols
    south_dist = (food_row - player_row) % num_rows
    west_dist = (-food_col + player_col) % num_cols

    distance_array = [north_dist, east_dist, south_dist, west_dist]
    optimal_move = min(distance_array)

    if player_index == 0:
        print("Distance Array (NESW): " + str(distance_array))
        print("Valid Moves: " + str(valid_moves))
    x = None
    if distance_array.index(optimal_move) == 0 and "NORTH" in valid_moves:
        x = Action.NORTH.name
    if distance_array.index(optimal_move) == 1 and "EAST" in valid_moves:
        x = Action.EAST.name
    if distance_array.index(optimal_move) == 2 and "SOUTH" in valid_moves:
        x = Action.SOUTH.name
    if distance_array.index(optimal_move) == 3 and "WEST" in valid_moves:
        x = Action.WEST.name

    if valid_moves == []:
        x = last_move[player_index]
    if x == None:
        x = random.choice(valid_moves)
    last_move[player_index] = x

    if player_index == 0:
        print("Move Taken")
        print(x)

    return x


# simple testing output for diagnosing issues with helper functions


def print_diagnostics(
    player_index, agent_positions, food_positions, near_food, potential_moves
):
    print("diagnostics---------------------------")
    print("current agent")
    print(player_index)

    print("new agent positions")
    print(agent_positions)

    print("new food positions")
    print(food_positions)

    print("nearest food position")
    print(near_food)

    print("potential moves")
    print(potential_moves)


# returns a representation of the game board as an array of various integers


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

    return np.asfarray(table, float)


# kaggle will run a simulation with the below function's output over the period of a round.
# This function brings everything together and produces a move to be accepted by the kaggle env.


def agent(obs_dict, config_dict):

    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index

    agent_positions = get_all_geese(observation)
    food_positions = get_all_food(observation)
    near_food = get_nearest_food(agent_positions, food_positions, player_index)
    potential_moves = get_potential_moves(agent_positions)
    valid_moves = get_valid_moves(agent_positions, potential_moves, player_index)
    if player_index == 0:
        print(
            get_table(
                player_index, agent_positions, food_positions, near_food, valid_moves
            )
        )

    return get_move(
        agent_positions,
        food_positions,
        potential_moves,
        valid_moves,
        near_food,
        player_index,
    )


env = make("hungry_geese")


#%%
# ensuring GPU is available to perform calculations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(device)

# implementing ML model


class DQN(nn.Module):
    def __init__(self, n_acts):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(77, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 16), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(16, 4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(4, n_acts))

    def forward(self, obs):
        q_values = self.layer1(obs)
        q_values = self.layer2(q_values)
        q_values = q_values.view(-1, 16)
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


# implementing experience replay


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


# instantiating hyper-parameters
n_episodes = 100000
train_batch_size = 32
print_freq = 100
learning_rate = 2.5e-4
target_update_delay = 100
update_freq = 32
global_step = 0
max_steps = 200
epsilon = 0

# book-keeping variables
done = False
n_acts = 4
reward = 0
prev_reward = 0
all_rewards = []

# instantiating experience replay
er_capacity = 50000
er = ExperienceReplay(er_capacity)

# instantiating model & target model
model = DQN(n_acts=n_acts).cuda()
target_model = copy.deepcopy(model)
target_model.to(device)

# instantiating optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)

# instantiating training agent
trainer = env.train([None, agent, agent, agent])
obs = trainer.reset()

# start training algorithm
for episode in range(n_episodes):
    # reset reward
    episode_reward = 0
    true_reward = 0
    prev_reward = 0
    reward = 0
    done = False
    while not done:
        # get table to feed into model
        agent_positions = get_all_geese(obs)
        food_positions = get_all_food(obs)
        near_food = get_nearest_food(agent_positions, food_positions, 0)
        potential_moves = get_potential_moves(agent_positions)
        valid_moves = get_valid_moves(agent_positions, potential_moves, 0)
        player_index = 0
        table = get_table(
            player_index, agent_positions, food_positions, near_food, valid_moves
        )
        table = table.reshape(77)

        # choosing a move
        if np.random.rand() < epsilon:
            act = np.random.choice(range(n_acts))
        else:
            obs_tensor = torch.tensor([table]).float().to(device)
            q_values = model(obs_tensor)[0]
            q_values = q_values.cpu().detach().numpy()
            act = np.argmax(q_values)

        act_num = act

        if act == 0:
            act = "NORTH"
        if act == 1:
            act = "EAST"
        if act == 2:
            act = "SOUTH"
        if act == 3:
            act = "WEST"

        # setting previous reward, and stepping the agent
        prev_reward = reward
        next_obs, reward, done, info = trainer.step(act)

        # setting the true (custom reward) so as to properly train the model
        if next_obs.step >= 3:
            if reward > prev_reward:
                true_reward = 100
            if reward == prev_reward:
                true_reward = 1
            if reward < prev_reward:
                true_reward = 0
        episode_reward += true_reward

        # if the round is over, the env is reset and the loop breaks
        if done:
            obs = trainer.reset()
            break

        # getting next board state to feed into experience replay
        agent_positions = get_all_geese(next_obs)
        food_positions = get_all_food(next_obs)
        near_food = get_nearest_food(agent_positions, food_positions, 0)
        potential_moves = get_potential_moves(agent_positions)
        valid_moves = get_valid_moves(agent_positions, potential_moves, 0)
        player_index = 0
        next_table = get_table(
            player_index, agent_positions, food_positions, near_food, valid_moves
        )
        next_table = next_table.reshape(77)
        er.add_step([table, act_num, reward, next_table, int(done)])
        obs = next_obs  # for next run
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

        global_step += 1
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