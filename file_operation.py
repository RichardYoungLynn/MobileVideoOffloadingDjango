import math
import random

local_lines_train = open("data/train/train_local.txt", "r").readlines()
server_lines_train = open("data/train/train_server.txt", "r").readlines()

local_lines_test = open("data/test/test_local.txt", "r").readlines()
server_lines_test = open("data/test/test_server.txt", "r").readlines()


def readLocalReward(index, train):
    if train == 1:
        line = local_lines_train[index].strip()
    elif train == 0:
        line = local_lines_test[index].strip()
    rewards = line.split(' ')
    result = {'index': rewards[0], 'local_people_num': rewards[1], 'file_size': rewards[2], 'memory_usage': rewards[3],
              'cpu_usage': rewards[4], 'local_confidence_sum': rewards[5], 'local_process_time': rewards[6]}
    return result


def readServerReward(index, train):
    if train == 1:
        line = server_lines_train[index].strip()
    elif train == 0:
        line = server_lines_test[index].strip()
    rewards = line.split(' ')
    result = {'index': rewards[0], 'server_people_num': rewards[1], 'server_confidence_sum': rewards[2],
              "server_process_time": rewards[3],"server_transmission_time_selftest": rewards[4],
              "server_transmission_time_4g": rewards[5],"server_transmission_time_5g": rewards[6]}
    return result


def LayeringReward(num):
    return math.ceil(float(num) / 0.1)


def TestTrainEnvLayered():
    train = 1
    train_len = 600
    test_len = 600
    if train == 0:
        len1 = test_len
    else:
        len1 = train_len
    local = 0
    server = 0
    random_num = 0
    local_reward_sum = 0.0
    server_reward_sum = 0.0
    random_reward_sum = 0.0
    max_reward = 0.0
    local_r1_list = []
    local_r2_list = []
    server_r1_list = []
    server_r2_list = []
    for i in range(len1):
        local_people_num = float(readLocalReward(i, train)['local_people_num'])
        local_confidence_sum = float(readLocalReward(i, train)['local_confidence_sum'])
        local_process_time = float(readLocalReward(i, train)['local_process_time'])
        memory_usage = float(readLocalReward(i, train)['memory_usage'])
        cpu_usage = float(readLocalReward(i, train)['cpu_usage'])

        server_people_num = float(readServerReward(i, train)['server_people_num'])
        server_confidence_sum = float(readServerReward(i, train)['server_confidence_sum'])
        server_process_time = float(readServerReward(i, train)['server_process_time'])
        server_transmission_time_selftest = float(readServerReward(i, train)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readServerReward(i, train)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readServerReward(i, train)['server_transmission_time_5g'])

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

        # local_r1_list.append(local_r1)
        # local_r2_list.append(local_r2)
        # server_r1_list.append(server_r1)
        # server_r2_list.append(server_r2)

        # local_r1 = (local_r1 - 0) / (30.350634602726156 - 0)
        # server_r1 = (server_r1 - 0) / (7.183713500529839 - 0)
        # local_r2 = (local_r2 - 0.354739294085071) / (1.6258874023744248 - 0.354739294085071)
        # server_r2 = (server_r2 - 0.08862650729817263) / (0.34017524373138897 - 0.08862650729817263)

        r1_sum = local_r1 + server_r1
        local_r1 = local_r1 / r1_sum
        server_r1 = server_r1 / r1_sum
        r2_sum = local_r2 + server_r2
        local_r2 = local_r2 / r2_sum
        server_r2 = server_r2 / r2_sum

        local_reward = local_r1 - local_r2
        local_reward_sum += local_reward
        server_reward = server_r1 - server_r2
        server_reward_sum += server_reward

        offload = local_reward < server_reward
        random_policy = random.choice([0, 1])
        random_reward = 0

        if offload:
            if offload == random_policy:
                random_num += 1
                random_reward = server_reward
            else:
                random_reward = local_reward
            server += 1
            max_reward += server_reward
        else:
            if offload == random_policy:
                random_num += 1
                random_reward = local_reward
            else:
                random_reward = server_reward
            local += 1
            max_reward += local_reward

        random_reward_sum += random_reward
        print("index = " + str(i) +
              ", random_reward = " + str(random_reward) + ", local_reward = " + str(
            local_reward) + ", server_reward = " + str(server_reward) +
              ", local_r1 = " + str(local_r1) + ", local_r2 = " + str(local_r2) +
              ", server_r1 = " + str(server_r1) + ", server_r2 = " + str(server_r2) +
              ", offload = " + str(local_reward < server_reward))

    print("random = " + str(random_num) + ", local = " + str(local) + ", server = " + str(server) +
          ", max_reward = " + str(max_reward) +
          ", random_reward_sum = " + str(random_reward_sum) + ", local_reward_sum = " + str(
        local_reward_sum) + ", server_reward_sum = " + str(server_reward_sum))

    # print("max_local_r1 = "+str(max(local_r1_list))+", min_local_r1 = "+str(min(local_r1_list))+
    #       ", max_local_r2 = "+str(max(local_r2_list))+", min_local_r2 = "+str(min(local_r2_list))+
    #       ", max_server_r1 = "+str(max(server_r1_list))+", min_server_r1 = "+str(min(server_r1_list))+
    #       ", max_server_r2 = "+str(max(server_r2_list))+", min_server_r2 = "+str(min(server_r2_list)))


if __name__ == '__main__':
    # print(readLocalReward(0,1))
    # print(readLocalReward(0,0))
    # print(readServerReward(0,1))
    # print(readServerReward(0,0))
    # sortByPeopleNum()
    # TestVideoOffloadEnv()
    # TestTrainEnv()
    TestTrainEnvLayered()
