'''Observation:
        Type: Box(2)
        Num	Observation               Min           Max
        0	PeopleNum                 0.0           20.0
        1	FileSize                  1000000.0     9999999.0

Actions:
        Type: Discrete(2)
        Num	Action
        0	local
        1	offload'''

from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math

class VideoOffloadEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 1000000.0]), np.array([20.0, 9999999.0]))
        self.count = 0

    def step(self, action, train):
        if action == 0:
            local_people_num=float(readLocalReward(self.count, train)['local_people_num'])
            local_people_confidence_sum=float(readLocalReward(self.count, train)['local_people_confidence_sum'])
            local_process_time=float(readLocalReward(self.count, train)['local_process_time'])
            # reward=(local_people_num*local_people_confidence_sum)/math.exp(local_process_time)
        elif action == 1:
            server_people_num=float(readServerReward(self.count, train)['server_people_num'])
            server_people_confidence_sum = float(readServerReward(self.count, train)['server_people_confidence_sum'])
            server_process_time = float(readServerReward(self.count, train)['server_process_time'])
            server_transmission_time_selftest = float(readServerReward(self.count, train)['server_transmission_time_selftest'])
            server_transmission_time_4g = float(readServerReward(self.count, train)['server_transmission_time_4g'])
            server_transmission_time_5g = float(readServerReward(self.count, train)['server_transmission_time_5g'])
            # reward=(server_people_num*server_people_confidence_sum)/math.exp(server_process_and_transmission_time)

        done = False
        if train==1:
            if (self.count == 927):
                done = True
            else:
                self.count += 1
        elif train==0:
            if (self.count == 299):
                done = True
            else:
                self.count += 1

        state = np.array([readLocalReward(self.count, train)['local_people_num'],
                          readLocalReward(self.count, train)['file_size']])

        return state, reward, done, {}

    def reset(self, train):
        self.count=0
        state = np.array([readLocalReward(self.count, train)['local_people_num'],
                          readLocalReward(self.count, train)['bandwidth_quality']])
        return state

    # def render(self, mode='human'):
    #     return None
    #
    # def close(self):
    #     return None

if __name__ == '__main__':
    env = VideoOffloadEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)