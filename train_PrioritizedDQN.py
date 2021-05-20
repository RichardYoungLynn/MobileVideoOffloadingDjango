#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parl.utils import check_version_for_fluid  # requires parl >= 1.4.1
check_version_for_fluid()

import argparse
import os
import time
import math
import pickle
from collections import deque
from datetime import datetime

import gym
import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm

import parl
from PrioritizedDQN.atari_agent import AtariAgent
from PrioritizedDQN.atari_model import AtariModel
from parl.utils import logger
from PrioritizedDQN.per_alg import PrioritizedDoubleDQN, PrioritizedDQN
from PrioritizedDQN.proportional_per import ProportionalPER
from train_env import TrainEnv

MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 20000
UPDATE_FREQ = 5
GAMMA = 0.99
LEARNING_RATE = 0.0005


def beta_adder(init_beta, step_size=0.0001):
    beta = init_beta
    step_size = step_size

    def adder():
        nonlocal beta, step_size
        beta += step_size
        return min(beta, 1)

    return adder


def process_transitions(transitions):
    transitions = np.array(transitions)
    batch_obs = np.stack(transitions[:, 0].copy())
    batch_act = transitions[:, 1].copy()
    batch_reward = transitions[:, 2].copy()
    batch_next_obs = np.stack(transitions[:, 3]).copy()
    batch_terminal = transitions[:, 4].copy()
    batch = (batch_obs, batch_act, batch_reward, batch_next_obs,
             batch_terminal)
    return batch


def run_episode(env, agent, per, mem=None, warmup=False, train=False):
    total_reward = 0
    obs = env.reset(1)
    steps = 0
    if warmup:
        decay_exploration = False
    else:
        decay_exploration = True
    while True:
        steps += 1
        action = agent.sample(obs, decay_exploration=decay_exploration)
        next_obs, reward, terminal, _ = env.step(action, 1)
        transition = [obs, action, reward, next_obs, terminal]
        if warmup:
            mem.append(transition)
        if train:
            per.store(transition)
            if steps % UPDATE_FREQ == 0:
                beta = get_beta()
                transitions, idxs, sample_weights = per.sample(beta=beta)
                batch = process_transitions(transitions)

                cost, delta = agent.learn(*batch, sample_weights)
                per.update(idxs, delta)

        total_reward += reward
        obs = next_obs
        if terminal:
            break

    return total_reward, steps


def run_evaluate_episode(env, agent):
    obs = env.reset(0)
    total_reward = 0
    while True:
        action = agent.predict(obs)
        obs, reward, isOver, info = env.step(action, 0)
        total_reward += reward
        if isOver:
            break
    return total_reward


def main():
    env = TrainEnv()

    per = ProportionalPER(alpha=0.6, seg_num=args.batch_size, size=MEMORY_SIZE)

    act_dim = env.action_space.n
    obs_shape = env.observation_space.shape
    model = AtariModel(act_dim)
    if args.alg == 'ddqn':
        algorithm = PrioritizedDoubleDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    elif args.alg == 'dqn':
        algorithm = PrioritizedDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = AtariAgent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=act_dim)

    # Replay memory warmup
    total_step = 0
    mem = []
    while total_step < MEMORY_WARMUP_SIZE:
        total_reward, steps = run_episode(
            env, agent, per, mem=mem, warmup=True)
        total_step += steps
    per.elements.from_list(mem[:int(MEMORY_WARMUP_SIZE)])

    pbar = tqdm(total=args.max_episode)

    if os.path.exists('prioritized_dqn_model'):
        agent.restore('./prioritized_dqn_model')
        print("加载模型成功，开始预测：")
        run_evaluate_episode(env, agent)

    log_list = []
    fo = open("log/" + str(math.floor(time.time() * 1000.0)) + "prioritized_dqn.txt", "w")
    train_episode = 0
    test_episode = 0

    while train_episode < args.max_episode:
        for i in range(0, 10):
            total_reward, steps = run_episode(env, agent, per, train=True)
            train_episode += 1
            pbar.update(1)
            log_list.append("Train " + str(train_episode) + " " + str(total_reward) + "\n")
            logger.info('train_episode:{}    train_reward:{}'.format(train_episode, total_reward))

        eval_reward = run_evaluate_episode(env, agent)
        log_list.append("Test " + str(test_episode) + " " + str(eval_reward) + "\n")
        logger.info('test_episode:{}    test_reward:{}'.format(test_episode, eval_reward))
        test_episode += 1

    fo.writelines(log_list)
    fo.close()
    pbar.close()

    agent.save('./prioritized_dqn_model')
    print("模型保存成功")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument(
        '--alg',
        type=str,
        default="dqn",
        help='dqn or ddqn, training algorithm to use.')
    parser.add_argument(
        '--max_episode',
        type=int,
        default=int(5000),
        help='maximum environmental episodes of games')
    args = parser.parse_args()
    assert args.alg in ['dqn','ddqn'], \
        'used algorithm should be dqn or ddqn (double dqn)'
    get_beta = beta_adder(init_beta=0.5)
    main()