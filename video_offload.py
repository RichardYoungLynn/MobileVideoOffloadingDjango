'''
Observation:
        Type: Box(2)
        Num     Observation           Min           Max           Note
        0	    PeopleNum             0.0           20.0          Discrete (0, 1, 2, ... , 20)
        1	    FileSize              1000000.0     9999999.0     Continuous
        2       MemoryUsage           0.0           1.0           Continuous
        3       CPUUsage              0.0           1.0           Continuous
        # 4       CPUFreq               1.1           2.2           Discrete (1.1, 1.4, 1.7, 1.9, 2.1, 2.2)
Actions:
        Type: Discrete(2)
        Num	    Action
        0	    local
        1	    offload
Reward:
        local_power_comsumption = 200mAH
        local_reward = local_people_confidence_sum / math.exp(local_process_time) - (memory_usage + cpu_usage) / 2 * math.exp((local_power_comsumption / 3600) * local_process_time)
        server_reward = server_people_confidence_sum / math.exp(server_process_time + server_transmission_time)
        reward = local_reward + server_reward
'''

from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math

class VideoOffloadEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(np.array([0.0, 1000000.0, 0.0, 0.0]), np.array([20.0, 9999999.0, 1.0, 1.0]))
        # self.observation_space = spaces.Box(np.array([0.0]), np.array([30.0]))
        # self.observation_space = spaces.Box(np.array([0.0, 1000000.0]), np.array([20.0, 9999999.0]))
        # 文件大小转为MB blow
        self.observation_space = spaces.Box(np.array([0.0, 0.11, 0.0, 0.0]), np.array([20.0, 1.2, 1.0, 1.0]))
        self.count = 0


    def step(self, action, train):
        local_confidence_sum = float(readLocalReward(self.count, train)['local_confidence_sum'])
        local_process_time = float(readLocalReward(self.count, train)['local_process_time'])
        memory_usage = float(readLocalReward(self.count, train)['memory_usage'])
        cpu_usage = float(readLocalReward(self.count, train)['cpu_usage'])
        r1 = local_confidence_sum / local_process_time
        r2 = math.exp((memory_usage + cpu_usage) * 300 * local_process_time * 0.001)
        local_reward = r1 - r2

        server_confidence_sum = float(readServerReward(self.count, train)['server_confidence_sum'])
        server_process_time = float(readServerReward(self.count, train)['server_process_time'])
        server_transmission_time_selftest = float(
            readServerReward(self.count, train)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readServerReward(self.count, train)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readServerReward(self.count, train)['server_transmission_time_5g'])
        server_reward = server_confidence_sum / (server_process_time + server_transmission_time_selftest)

        # local_people_num = float(readLocalReward(self.count, train)['local_people_num'])
        # file_size = float(readLocalReward(self.count, train)['file_size'])

        # offload = local_people_num > 10

        offload = local_reward < server_reward

        if train==1:
            if (self.count == 891):
                reward = 0
                done = True
            else:
                self.count += 1
                if action == 0:
                    if offload:
                        reward = 0
                        done = True
                    else:
                        reward = 1
                        done = False
                elif action == 1:
                    if offload:
                        reward = 1
                        done = False
                    else:
                        reward = 0
                        done = True
        elif train==0:
            if (self.count == 320):
                reward = 0
                done = True
            else:
                self.count += 1
                if action == 0:
                    if offload:
                        reward = 0
                        done = True
                    else:
                        reward = 1
                        done = False
                elif action == 1:
                    if offload:
                        reward = 1
                        done = False
                    else:
                        reward = 0
                        done = True

        state = np.array([readLocalReward(self.count, train)['local_people_num'],readLocalReward(self.count, train)['file_size'],
                          readLocalReward(self.count, train)['memory_usage'],readLocalReward(self.count, train)['cpu_usage']])

        # state = np.array([readLocalReward(self.count, train)['local_people_num']])

        # state = np.array([readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size']])

        return state, reward, done, {}


    def reset(self, train):
        self.count=0
        self.record = []
        state = np.array([readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size'],
                          readLocalReward(self.count, train)['memory_usage'], readLocalReward(self.count, train)['cpu_usage']])

        # state = np.array([readLocalReward(self.count, train)['local_people_num']])

        # state = np.array([readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size']])
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