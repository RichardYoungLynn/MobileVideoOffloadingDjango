'''
Observation:
        Type: Box(2)
        Num     Observation           Min           Max           Note
        0	    PeopleNum             0.0           50.0          Discrete
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
        server_reward = server_people_num / server_process_time + server_transmission_time
'''

from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math


class TrainEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, 0.0, 0.0]), np.array([20.0, 1.0, 1.0, 1.0]))
        self.count = 0

    # def cal_reward(self, offload, action, train):
    #     done = False
    #     if train == 1:
    #         if (self.count == 599):
    #             if action == 0:
    #                 if offload:
    #                     reward = 0
    #                 else:
    #                     reward = 1
    #             elif action == 1:
    #                 if offload:
    #                     reward = 1
    #                 else:
    #                     reward = 0
    #             done = True
    #         else:
    #             self.count += 1
    #             if action == 0:
    #                 if offload:
    #                     reward = 0
    #                 else:
    #                     reward = 1
    #             elif action == 1:
    #                 if offload:
    #                     reward = 1
    #                 else:
    #                     reward = 0
    #     elif train == 0:
    #         if (self.count == 599):
    #             if action == 0:
    #                 if offload:
    #                     reward = 0
    #                 else:
    #                     reward = 1
    #             elif action == 1:
    #                 if offload:
    #                     reward = 1
    #                 else:
    #                     reward = 0
    #             done = True
    #         else:
    #             self.count += 1
    #             if action == 0:
    #                 if offload:
    #                     reward = 0
    #                 else:
    #                     reward = 1
    #             elif action == 1:
    #                 if offload:
    #                     reward = 1
    #                 else:
    #                     reward = 0
    #     return done, reward

    def cal_reward(self, action, train, local_reward, server_reward):
        done = False
        if train == 1:
            if (self.count == 599):
                if action == 0:
                    reward = local_reward
                elif action == 1:
                    reward = server_reward
                done = True
            else:
                self.count += 1
                if action == 0:
                    reward = local_reward
                elif action == 1:
                    reward = server_reward
        elif train == 0:
            if (self.count == 599):
                if action == 0:
                    reward = local_reward
                elif action == 1:
                    reward = server_reward
                done = True
            else:
                self.count += 1
                if action == 0:
                    reward = local_reward
                elif action == 1:
                    reward = server_reward
        return done, reward

    def step(self, action, train):
        local_people_num = float(readLocalReward(self.count, train)['local_people_num'])
        local_confidence_sum = float(readLocalReward(self.count, train)['local_confidence_sum'])
        local_process_time = float(readLocalReward(self.count, train)['local_process_time'])
        memory_usage = float(readLocalReward(self.count, train)['memory_usage'])
        cpu_usage = float(readLocalReward(self.count, train)['cpu_usage'])

        server_people_num = float(readServerReward(self.count, train)['server_people_num'])
        server_confidence_sum = float(readServerReward(self.count, train)['server_confidence_sum'])
        server_process_time = float(readServerReward(self.count, train)['server_process_time'])
        server_transmission_time_selftest = float(
            readServerReward(self.count, train)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readServerReward(self.count, train)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readServerReward(self.count, train)['server_transmission_time_5g'])

        if local_people_num == 0:
            local_r1 = 0
        else:
            local_r1 = local_confidence_sum / math.log(local_process_time)
        local_r2 = memory_usage + cpu_usage

        if server_people_num == 0:
            server_r1 = 0
        else:
            server_r1 = server_confidence_sum / math.log(server_process_time)
        server_r2 = server_transmission_time_selftest

        r1_sum = local_r1 + server_r1
        local_r1 = local_r1 / r1_sum
        server_r1 = server_r1 / r1_sum
        r2_sum = local_r2 + server_r2
        local_r2 = local_r2 / r2_sum
        server_r2 = server_r2 / r2_sum

        local_reward = local_r1 - local_r2
        server_reward = server_r1 - server_r2

        offload = local_reward < server_reward

        # done, reward = self.cal_reward(offload, action, train)
        done, reward = self.cal_reward(action, train, local_reward, server_reward)

        state = np.array(
            [readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size'],
             readLocalReward(self.count, train)['memory_usage'], readLocalReward(self.count, train)['cpu_usage']])

        return state, reward, done, {}

    def reset(self, train):
        self.count = 0

        state = np.array(
            [readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size'],
             readLocalReward(self.count, train)['memory_usage'], readLocalReward(self.count, train)['cpu_usage']])

        return state


if __name__ == '__main__':
    pass
