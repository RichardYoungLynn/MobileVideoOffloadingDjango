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
        local_reward = local_people_num / local_process_time - math.exp(memory_usage + cpu_usage)
        server_reward = server_people_num / (server_process_time + server_transmission_time)
'''

from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math


class AnalysisEnv():

    def __init__(self, type):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, 0.0, 0.0]), np.array([30.0, 1.0, 1.0, 1.0]))
        self.count = 0
        self.type = type

    def LayeringReward(self, num):
        return math.ceil(float(num) / 0.1)

    def ReadLocalReward(self):
        lines = open("data/layering/analysis/" + self.type + "/local.txt", "r").readlines()
        line = lines[self.count].strip()
        rewards = line.split(' ')
        result = {'local_people_num': rewards[0], 'file_size': rewards[1], 'memory_usage': rewards[2],
                  'cpu_usage': rewards[3], 'local_process_time': rewards[4]}
        return result

    def ReadServerReward(self):
        lines = open("data/layering/analysis/" + self.type + "/server.txt", "r").readlines()
        line = lines[self.count].strip()
        rewards = line.split(' ')
        result = {'server_people_num': rewards[0], 'server_process_time': rewards[1],
                  'server_transmission_time_selftest': rewards[2],
                  'server_transmission_time_4g': rewards[3], 'server_transmission_time_5g': rewards[4]}
        return result

    def step(self, action):
        local_people_num = float(self.ReadLocalReward()['local_people_num'])
        local_process_time = float(self.ReadLocalReward()['local_process_time'])
        memory_usage = float(self.ReadLocalReward()['memory_usage'])
        cpu_usage = float(self.ReadLocalReward()['cpu_usage'])
        r1 = local_people_num / local_process_time
        r2 = math.exp(memory_usage + cpu_usage)
        local_reward = r1 - r2

        server_people_num = float(self.ReadServerReward()['server_people_num'])
        server_process_time = float(self.ReadServerReward()['server_process_time'])
        server_transmission_time_selftest = float(self.ReadServerReward()['server_transmission_time_selftest'])
        server_transmission_time_4g = float(self.ReadServerReward()['server_transmission_time_4g'])
        server_transmission_time_5g = float(self.ReadServerReward()['server_transmission_time_5g'])
        server_reward = server_people_num / (server_process_time + server_transmission_time_selftest)

        offload = local_reward < server_reward

        if action == 0:
            evaluate_reward = local_reward
        else:
            evaluate_reward = server_reward

        print("correct = " + str((offload + 0) == action) + ", action = " + str(action) + ", offload = " + str(offload) + ", local_reward = " + str(local_reward) + ", server_reward = " + str(server_reward) + ", reward = " + str(evaluate_reward))

        done = False
        if (self.count == 99):
            done = True
        else:
            self.count += 1

        state = np.array([self.ReadLocalReward()['local_people_num'],
                          self.ReadLocalReward()['file_size'],
                          self.ReadLocalReward()['memory_usage'],
                          self.ReadLocalReward()['cpu_usage']])

        return state, evaluate_reward, local_reward, server_reward, done, {}

    def reset(self, type):
        self.count = 0
        self.type = type

        state = np.array([self.ReadLocalReward()['local_people_num'],
                          self.ReadLocalReward()['file_size'],
                          self.ReadLocalReward()['memory_usage'],
                          self.ReadLocalReward()['cpu_usage']])

        return state


if __name__ == '__main__':
    pass
