# -*- coding: utf-8 -*-
#%%

import random
import numpy as np
import pandas as pd
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
env.run([agent, agent, agent, agent])
env.render(mode="ipython")
