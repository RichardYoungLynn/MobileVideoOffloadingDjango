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
from parl.utils import logger

from video_offload import VideoOffloadEnv

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def run_episode(env):
    total_reward = 0
    env.reset(1)
    step = 0
    while True:
        step += 1
        next_obs, reward, isOver, _ = env.step(0,1)
        total_reward += reward
        if isOver:
            break
    return total_reward


def evaluate(env):
    eval_reward = []
    for i in range(5):
        env.reset(0)
        episode_reward = 0
        isOver = False
        while not isOver:
            obs, reward, isOver, _ = env.step(0,0)
            episode_reward += reward
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)



def main():
    env = VideoOffloadEnv()
    max_episode = 20
    # start train
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(0, 5):
            total_reward = run_episode(env)
            episode += 1
            logger.info('episode:{}    total_reward:{}'.format(episode, total_reward))

        eval_reward = evaluate(env)
        logger.info('episode:{}    test_reward:{}'.format(episode, eval_reward))


if __name__ == '__main__':
    main()