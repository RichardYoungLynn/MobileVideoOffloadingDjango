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

import numpy as np
import os
import time
import math
import parl
from parl.utils import logger

from DQN.cartpole_model import CartpoleModel
from DQN.cartpole_agent import CartpoleAgent
from DQN.replay_memory import ReplayMemory
from train_env import TrainEnv

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def run_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset(1)
    # logger.info('The obs in run_episode after reset is:{}'.format(obs))
    # just like [ 0.04565629 -0.00461014 -0.03107543 -0.01817521]

    step = 0
    while True:
        # logger.info('The step is:{}'.format(step))
        step += 1
        action = agent.sample(obs)
        # logger.info('The action is:{}'.format(action))
        next_obs, reward, isOver, _ = env.step(action,1)
        rpm.append((obs, action, reward, next_obs, isOver))
        # logger.info('The step function is:{}'.format(env.step(action)))
        # just like (array([ 0.01530615,  0.41110149, -0.03372429, -0.58779079]), 1.0, False, {})
        # logger.info('After step, next_obs is:{}    reward is:{}    isOver is:{}'.format(next_obs, reward, isOver))
        # just like After step, next_obs is:[ 0.01099432  0.2155917  -0.02799607 -0.28641124]    reward is:1.0    isOver is:False

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_isOver)
            # logger.info('The train loss is:{}'.format(train_loss))

        total_reward += reward
        obs = next_obs
        if isOver:
            break
    return total_reward


def evaluate(agent, env):
    # test part, run 5 episodes and average
    # eval_reward = []
    # for i in range(5):
    #     obs = env.reset(1)
    #     episode_reward = 0
    #     isOver = False
    #     while not isOver:
    #         action = agent.predict(obs)
    #         obs, reward, isOver, _ = env.step(action, 1)
    #         episode_reward += reward
    #     eval_reward.append(episode_reward)
    # return np.mean(eval_reward)

    obs = env.reset(0)
    episode_reward = 0
    isOver = False
    while not isOver:
        action = agent.predict(obs)
        obs, reward, isOver, _ = env.step(action, 0)
        episode_reward += reward
    return episode_reward


def main():
    # env = gym.make('CartPole-v0')
    env = TrainEnv()
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape
    # logger.info('action_dim:{}    obs_shape:{}    obs_shape[0]:{}'.format(action_dim, obs_shape, obs_shape[0]))
    # just like action_dim:2    obs_shape:(4,)    obs_shape[0]:4

    rpm = ReplayMemory(MEMORY_SIZE)

    model = CartpoleModel(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = CartpoleAgent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim,
        e_greed=0.9,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training

    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm)

    if os.path.exists('dqn_model'):
        agent.restore('./dqn_model')
        print("????????????????????????????????????")
        evaluate(agent, env)

    max_episode = 2000

    log_list = []
    fo = open("log/"+str(math.floor(time.time()*1000.0))+"dqn.txt", "w")
    train_episode = 0
    test_episode=0
    while train_episode < max_episode:
        for i in range(0, 10):
            total_reward = run_episode(agent, env, rpm)
            train_episode += 1
            log_list.append("Train "+str(train_episode)+" "+str(total_reward)+"\n")
            logger.info('train_episode:{}    train_reward:{}'.format(train_episode, total_reward))

        eval_reward = evaluate(agent, env)
        log_list.append("Test "+str(test_episode) + " " + str(eval_reward) + "\n")
        logger.info('test_episode:{}    test_reward:{}'.format(test_episode, eval_reward))
        test_episode+=1

    fo.writelines(log_list)
    fo.close()

    agent.save('./dqn_model')
    print("??????????????????")


if __name__ == '__main__':
    main()