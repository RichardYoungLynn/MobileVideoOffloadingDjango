#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import gym
import time
import math
import paddle.fluid as fluid
import numpy as np
import os
import parl
from DQN_variant.atari_agent import AtariAgent
from DQN_variant.atari_model import AtariModel
from datetime import datetime
from DQN_variant.replay_memory import ReplayMemory
from parl.utils import summary, logger
from tqdm import tqdm
from train_env import TrainEnv

MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 200
UPDATE_FREQ = 5
GAMMA = 0.99
LEARNING_RATE = 0.0005


def run_train_episode(env, agent, rpm):
    total_reward = 0
    all_cost = []
    obs = env.reset(1)
    steps = 0
    while True:
        steps += 1
        action = agent.sample(obs)
        next_obs, reward, isOver, _ = env.step(action, 1)
        rpm.append((obs, action, reward, next_obs, isOver))
        if len(rpm) > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                 batch_isOver) = rpm.sample(args.batch_size)
                cost = agent.learn(batch_obs, batch_action, batch_reward,
                                   batch_next_obs, batch_isOver)
                all_cost.append(float(cost))
        total_reward += reward
        obs = next_obs
        if isOver:
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
    act_dim = env.action_space.n
    obs_shape = env.observation_space.shape
    rpm = ReplayMemory(MEMORY_SIZE)

    model = AtariModel(act_dim, args.algo)
    if args.algo == 'DDQN':
        algorithm = parl.algorithms.DDQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    elif args.algo in ['DQN', 'Dueling']:
        algorithm = parl.algorithms.DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = AtariAgent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=act_dim)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        total_reward, steps = run_train_episode(env, agent, rpm)

    pbar = tqdm(total=args.max_episode)

    if os.path.exists('dqn_model'):
        agent.restore('./dqn_model')
        print("加载模型成功，开始预测：")
        run_evaluate_episode(env, agent)

    log_list = []
    fo = open("log/" + str(math.floor(time.time() * 1000.0)) + "dueling.txt", "w")
    train_episode = 0
    test_episode = 0
    while train_episode < args.max_episode:
        for i in range(0, 10):
            total_reward, steps = run_train_episode(env, agent, rpm)
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

    agent.save('./dqn_model')
    print("模型保存成功")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument(
        '--algo',
        default='DQN',
        help=
        'DQN/DDQN/Dueling, represent DQN, double DQN, and dueling DQN respectively',
    )
    parser.add_argument(
        '--max_episode',
        type=int,
        default=int(5000),
        help='maximum environmental episodes of games')

    args = parser.parse_args()
    main()
