import math
import random
import numpy as np
import matplotlib.pyplot as plt

lines_train = open("data/analysis/test.txt", "r").readlines()


def readReward(index):
    line = lines_train[index].strip()
    rewards = line.split(' ')
    result = {'local_index': rewards[0], 'local_people_num': rewards[1], 'file_size': rewards[2],
              'memory_usage': rewards[3],'cpu_usage': rewards[4],
              'local_confidence_sum': rewards[5], 'local_process_time': rewards[6],
              'server_index': rewards[7], 'server_people_num': rewards[8], 'server_confidence_sum': rewards[9],
              'server_process_time': rewards[10],'server_transmission_time_selftest': rewards[11],
              'server_transmission_time_4g': rewards[12],'server_transmission_time_5g': rewards[13]
              }
    return result


if __name__ == '__main__':
    len1 = 600
    x_list = []
    local_reward_list = []
    server_reward_list = []
    for i in range(len1):
        local_people_num = float(readReward(i)['local_people_num'])
        local_confidence_sum = float(readReward(i)['local_confidence_sum'])
        local_process_time = float(readReward(i)['local_process_time'])
        memory_usage = float(readReward(i)['memory_usage'])
        cpu_usage = float(readReward(i)['cpu_usage'])

        server_people_num = float(readReward(i)['server_people_num'])
        server_confidence_sum = float(readReward(i)['server_confidence_sum'])
        server_process_time = float(readReward(i)['server_process_time'])
        server_transmission_time_selftest = float(readReward(i)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readReward(i)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readReward(i)['server_transmission_time_5g'])

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

        x_list.append(i)
        local_reward_list.append(local_reward)
        server_reward_list.append(server_reward)

    x_np = np.asarray(x_list)
    local_reward_np = np.asarray(local_reward_list)
    server_reward_np = np.asarray(server_reward_list)

    plt.title("Result")
    plt.xlabel("index")
    plt.ylabel("reward")
    plt.plot(x_np, local_reward_np, color="red")
    plt.plot(x_np, server_reward_np, color="blue")
    plt.legend(('local reward', 'server reward'), loc='best')
    plt.show()
