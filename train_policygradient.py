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

#-*- coding: utf-8 -*-

import os
import gym
import time
import math
import numpy as np
import parl

from PolicyGradient.agent import Agent
from PolicyGradient.model import Model
from PolicyGradient.algorithm import PolicyGradient  # from parl.algorithms import PolicyGradient
from video_offload import VideoOffloadEnv
from parl.utils import logger
from train_env import TrainEnv

LEARNING_RATE = 1e-3


# 训练一个episode
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset(1)
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action,1)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def evaluate(env, agent):
    obs = env.reset(0)
    episode_reward = 0
    while True:
        action = agent.predict(obs)
        obs, reward, isOver, _ = env.step(action, 0)
        episode_reward += reward
        if isOver:
            break
    return episode_reward


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    # env = gym.make('CartPole-v0')
    # env = env.unwrapped # Cancel the minimum score limit
    # env = VideoOffloadEnv()
    env = TrainEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    if os.path.exists('./policygradient_model'):
        agent.restore('./policygradient_model')
        print("加载模型成功，开始预测：")
        evaluate(env, agent)

    max_episode = 20000

    log_list = []
    fo = open("log/" + str(math.floor(time.time() * 1000.0)) + "policygradient.txt", "w")
    train_episode = 0
    test_episode = 0
    while train_episode < max_episode:
        for i in range(0, 10):
            obs_list, action_list, reward_list = run_episode(env, agent)
            log_list.append("Train " + str(train_episode) + " " + str(sum(reward_list)) + "\n")
            logger.info("train_episode:{}    train_reward:{}.".format(train_episode, sum(reward_list)))

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list)

            agent.learn(batch_obs, batch_action, batch_reward)
            train_episode += 1

        total_reward = evaluate(env, agent)
        log_list.append("Test " + str(test_episode) + " " + str(total_reward) + "\n")
        logger.info('test_episode:{}    test_reward:{}'.format(test_episode, total_reward))
        test_episode += 1

    fo.writelines(log_list)
    fo.close()

    # save the parameters to ./policygradient_model
    agent.save('./policygradient_model')


if __name__ == '__main__':
    main()