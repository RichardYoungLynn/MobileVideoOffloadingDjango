from gym import spaces
from file_operation import readLocalReward, readServerReward
import numpy as np
import math

class TrainEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, 0.0, 0.0]), np.array([300.0, 100.0, 100.0, 100.0]))
        self.count = 0


    def cal_reward(self,offload,action,train):
        done = False
        if train == 1:
            if (self.count == 199):
                if action == 0:
                    if offload:
                        reward = 0
                    else:
                        reward = 1
                elif action == 1:
                    if offload:
                        reward = 1
                    else:
                        reward = 0
                done = True
            else:
                self.count += 1
                if action == 0:
                    if offload:
                        reward = 0
                    else:
                        reward = 1
                elif action == 1:
                    if offload:
                        reward = 1
                    else:
                        reward = 0
        elif train == 0:
            if (self.count == 99):
                if action == 0:
                    if offload:
                        reward = 0
                    else:
                        reward = 1
                elif action == 1:
                    if offload:
                        reward = 1
                    else:
                        reward = 0
                done = True
            else:
                self.count += 1
                if action == 0:
                    if offload:
                        reward = 0
                    else:
                        reward = 1
                elif action == 1:
                    if offload:
                        reward = 1
                    else:
                        reward = 0
        return done, reward


    def step(self, action, train):
        local_confidence_sum = float(readLocalReward(self.count, train)['local_confidence_sum'])
        local_process_time = float(readLocalReward(self.count, train)['local_process_time'])
        memory_usage = float(readLocalReward(self.count, train)['memory_usage'])
        cpu_usage = float(readLocalReward(self.count, train)['cpu_usage'])
        r1 = local_confidence_sum / local_process_time
        r2 = (memory_usage + cpu_usage) * local_process_time * 0.002
        local_reward = r1 - r2

        server_confidence_sum = float(readServerReward(self.count, train)['server_confidence_sum'])
        server_process_time = float(readServerReward(self.count, train)['server_process_time'])
        server_transmission_time_selftest = float(readServerReward(self.count, train)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readServerReward(self.count, train)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readServerReward(self.count, train)['server_transmission_time_5g'])
        server_reward = server_confidence_sum / (server_process_time + server_transmission_time_selftest)

        offload = local_reward < server_reward

        done, reward= self.cal_reward(offload, action, train)

        state = np.array([readLocalReward(self.count, train)['local_people_num'],readLocalReward(self.count, train)['file_size'],
                          readLocalReward(self.count, train)['memory_usage'],readLocalReward(self.count, train)['cpu_usage']])

        return state, reward, done, {}


    def reset(self, train):
        self.count=0

        state = np.array([readLocalReward(self.count, train)['local_people_num'], readLocalReward(self.count, train)['file_size'],
                          readLocalReward(self.count, train)['memory_usage'], readLocalReward(self.count, train)['cpu_usage']])

        return state


if __name__ == '__main__':
    pass