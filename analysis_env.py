'''
Observation:
        Type: Box(2)
        Num     Observation           Min           Max           Note
        0	    PeopleNum             0.0           30.0          Discrete
        1	    FileSize              0.0           1.0           Continuous
        2       MemoryUsage           0.0           1.0           Continuous
        3       CpuUsage              0.0           1.0           Continuous
Actions:
        Type: Discrete(2)
        Num	    Action
        0	    local
        1	    offload
Reward:
        local_reward = (local_people_num / local_process_time) - math.exp(memory_usage + cpu_usage)
        server_reward = server_people_num / (server_process_time + server_transmission_time)
'''

from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math


class AnalysisEnv():

    def __init__(self, type):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, 0.0, 0.0]), np.array([20.0, 1.0, 1.0, 1.0]))
        self.count = 0
        self.type = type

    def ReadLocalReward(self):
        lines = open("data/analysis/train.txt", "r").readlines()
        line = lines[self.count].strip()
        rewards = line.split(' ')
        result = {'index': rewards[0], 'local_people_num': rewards[1], 'file_size': rewards[2],'memory_usage': rewards[3],
                  'cpu_usage': rewards[4], 'local_confidence_sum': rewards[5], 'local_process_time': rewards[6]}
        return result

    def ReadServerReward(self):
        lines = open("data/analysis/train.txt", "r").readlines()
        line = lines[self.count].strip()
        rewards = line.split(' ')
        result = {'index': rewards[7], 'server_people_num': rewards[8], 'server_confidence_sum': rewards[9],
                  "server_process_time": rewards[10],"server_transmission_time_selftest": rewards[11],
                  "server_transmission_time_4g": rewards[12],"server_transmission_time_5g": rewards[13]}
        return result

    def step(self, action):
        local_people_num = float(self.ReadLocalReward()['local_people_num'])
        local_confidence_sum = float(self.ReadLocalReward()['local_confidence_sum'])
        local_process_time = float(self.ReadLocalReward()['local_process_time'])
        memory_usage = float(self.ReadLocalReward()['memory_usage'])
        cpu_usage = float(self.ReadLocalReward()['cpu_usage'])
        if local_people_num == 0:
            local_r1 = 0
        else:
            local_r1 = math.log(local_people_num + local_confidence_sum) / local_process_time
        local_r2 = math.exp(memory_usage + cpu_usage)
        local_reward = local_r1 - local_r2

        server_people_num = float(self.ReadServerReward()['server_people_num'])
        server_confidence_sum = float(self.ReadServerReward()['server_confidence_sum'])
        server_process_time = float(self.ReadServerReward()['server_process_time'])
        server_transmission_time_selftest = float(self.ReadServerReward()['server_transmission_time_selftest'])
        server_transmission_time_4g = float(self.ReadServerReward()['server_transmission_time_4g'])
        server_transmission_time_5g = float(self.ReadServerReward()['server_transmission_time_5g'])
        if server_people_num == 0:
            server_r1 = 0
        else:
            server_r1 = math.log(server_people_num + server_confidence_sum) / server_process_time
        server_r2 = math.exp(server_transmission_time_selftest)
        server_reward = server_r1 - server_r2

        offload = local_reward < server_reward

        if action == 0:
            evaluate_reward = local_reward
        else:
            evaluate_reward = server_reward

        # print("correct = " + str((offload + 0) == action) + ", action = " + str(action) + ", offload = " + str(offload) + ", local_reward = " + str(local_reward) + ", server_reward = " + str(server_reward) + ", reward = " + str(evaluate_reward))

        done = False
        if (self.count == 599):
            done = True
        else:
            self.count += 1

        state = np.array([self.ReadLocalReward()['local_confidence_sum'],
                          self.ReadLocalReward()['file_size'],
                          self.ReadLocalReward()['memory_usage'],
                          self.ReadLocalReward()['cpu_usage']])

        return state, evaluate_reward, local_reward, server_reward, done, {}

    def reset(self, type):
        self.count = 0
        self.type = type

        state = np.array([self.ReadLocalReward()['local_confidence_sum'],
                          self.ReadLocalReward()['file_size'],
                          self.ReadLocalReward()['memory_usage'],
                          self.ReadLocalReward()['cpu_usage']])

        return state


if __name__ == '__main__':
    pass
