import random
import math
import numpy as np
import os
import time
import math
import matplotlib
import matplotlib.pyplot as plt

import parl
from parl.utils import logger
from DQN_variant.atari_agent import AtariAgent
from DQN_variant.atari_model import AtariModel
from DQN_variant.replay_memory import ReplayMemory
from analysis_env import AnalysisEnv

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def ParseTrainResult(filename):
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    result_lines = open("data/result/" + filename + ".txt", "r").readlines()
    lines_len = len(result_lines)

    train_count = 0
    train_sum = 0
    test_count = 0
    test_sum = 0
    for index in range(lines_len):
        results = result_lines[index].strip().split(' ')
        if results[0] == 'Train':
            train_x_list.append(train_count)
            train_y_list.append(float(results[2]))
            train_count += 1
        if results[0] == 'Test':
            test_x_list.append(test_count)
            test_y_list.append(float(results[2]))
            test_count += 1

    print(len(train_x_list))
    print(len(train_y_list))

    train_x_np = np.asarray(train_x_list)
    train_dqn_np = np.asarray(train_y_list)
    train_local_np = np.linspace(290, 290, len(train_y_list))
    train_server_np = np.linspace(310, 310, len(train_y_list))
    train_random_np = np.linspace(312, 312, len(train_y_list))

    print(train_x_np)
    print(train_dqn_np)

    plt.title("Train result in Prioritized DQN")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(train_x_np, train_dqn_np, color="red")
    plt.plot(train_x_np, train_local_np, color="blue")
    plt.plot(train_x_np, train_server_np, color="yellow")
    plt.plot(train_x_np, train_random_np, color="green")
    plt.legend(('prioritized dqn', 'local', 'server', 'random'), loc='upper left')
    plt.savefig(fname="data/result/train_prioritized_dqn_analysis.png")
    plt.show()


def ParseTestResult(filename):
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    result_lines = open("data/result/" + filename + ".txt", "r").readlines()
    lines_len = len(result_lines)

    train_count = 0
    train_sum = 0
    test_count = 0
    test_sum = 0
    for index in range(lines_len):
        results = result_lines[index].strip().split(' ')
        if results[0] == 'Train':
            train_x_list.append(train_count)
            train_y_list.append(float(results[2]))
            train_count += 1
        if results[0] == 'Test':
            test_x_list.append(test_count)
            test_y_list.append(float(results[2]))
            test_count += 1

    print(len(test_x_list))
    print(len(test_y_list))

    test_x_np = np.asarray(test_x_list)
    test_dqn_np = np.asarray(test_y_list)
    test_local_np = np.linspace(289, 289, len(test_y_list))
    test_server_np = np.linspace(311, 311, len(test_y_list))
    test_random_np = np.linspace(296, 296, len(test_y_list))

    print(test_x_np)
    print(test_dqn_np)

    plt.title("Test result in Prioritized DQN")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(test_x_np, test_dqn_np, color="red")
    plt.plot(test_x_np, test_local_np, color="blue")
    plt.plot(test_x_np, test_server_np, color="yellow")
    plt.plot(test_x_np, test_random_np, color="green")
    plt.legend(('prioritized dqn', 'local', 'server', 'random'), loc='upper left')
    plt.savefig(fname="data/result/test_prioritized_dqn_analysis.png")
    plt.show()


def TakeSecond(elem):
    return float(elem[1])


def AddIndex(filename):
    list = []
    fo = open("data/layering/analysis/" + filename + ".txt", "r")
    lines = fo.readlines()
    fo.close()
    for index in range(len(lines)):
        line = str(index) + " " + lines[index].strip() + "\n"
        list.append(line)

    fo = open("data/layering/analysis/" + filename + ".txt", "w")
    fo.writelines(list)
    fo.close()


def CreateFile():
    list = []
    fo1 = open("data/layering/test/test_local_peoplenum_confsum.txt", "r")
    fo2 = open("data/layering/test/test_server_peoplenum_confsum.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    fo1.close()
    fo2.close()
    len1 = len(lines1)

    for index in range(len1):
        test_local_peoplenum = lines1[index].strip().split(' ')[0]
        test_server_peoplenum = lines2[index].strip().split(' ')[0]
        list.append(test_local_peoplenum + ' ' + test_server_peoplenum + '\n')

    fo = open("data/layering/analysis/local_people_num_analysis/local_server_peoplenum.txt", "w")
    fo.writelines(list)
    fo.close()


def SortFile():
    list = []
    fo1 = open("data/test/test_local.txt", "r")
    fo2 = open("data/test/test_server.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    fo1.close()
    fo2.close()
    len1 = len(lines1)
    for index in range(len1):
        line1 = lines1[index].strip().split(' ')
        line2 = lines2[index].strip().split(' ')
        list.append([line1[0], line1[1], line1[2], line1[3], line1[4], line1[5], line1[6],
                     line2[0], line2[1], line2[2], line2[3], line2[4], line2[5], line2[6]])

    list.sort(key=TakeSecond)

    lines = []
    for index in range(len1):
        line = list[index]
        lines.append(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] +' ' + line[4] + ' ' + line[5] + ' ' + line[6] + ' ' +
                     line[7] + ' ' + line[8] + ' ' + line[9] + ' ' + line[10] +' ' + line[11] + ' ' + line[12] + ' ' + line[13] + '\n')

    fo = open("data/analysis/test.txt", "w")
    fo.writelines(lines)
    fo.close()


def CreateNormalDistribution(filename, mu, sigma, num):
    list = []
    fo = open("data/layering/analysis/" + filename + ".txt", "w")
    s = np.random.normal(mu, sigma, num)
    s.sort()
    for index in range(len(s)):
        list.append(str(s[index]) + "\n")

    fo.writelines(list)
    fo.close()
    count, bins, ignored = plt.hist(s, 30, density=True, color='b')
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    plt.show()


def CreateMemCpuNormalDistribution(filename, mu, sigma, num):
    list = []
    fo = open("data/layering/analysis/" + filename + ".txt", "w")
    s1 = np.random.normal(mu, sigma, num)
    s2 = np.random.normal(mu, sigma, num)
    s1.sort()
    s2.sort()
    for index in range(len(s1)):
        list.append(str(s1[index]) + " " + str(s2[index]) + "\n")

    fo.writelines(list)
    fo.close()


def CreateDataset(analysis_type):
    list1 = []
    list2 = []
    fo1 = open("data/layering/analysis/" + analysis_type + "/file_size.txt", "r")
    fo2 = open("data/layering/analysis/" + analysis_type + "/local_server_peoplenum.txt", "r")
    fo3 = open("data/layering/analysis/" + analysis_type + "/local_protime.txt", "r")
    fo4 = open("data/layering/analysis/" + analysis_type + "/mem_cpu_usage.txt", "r")
    fo5 = open("data/layering/analysis/" + analysis_type + "/server_protime.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    lines3 = fo3.readlines()
    lines4 = fo4.readlines()
    lines5 = fo5.readlines()
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()
    fo5.close()
    # random.shuffle(lines1)
    random.shuffle(lines3)
    random.shuffle(lines4)
    random.shuffle(lines5)
    len1 = len(lines1)
    for index in range(len1):
        states1 = lines1[index].strip()
        states2 = lines2[index].strip().split(' ')
        states3 = lines3[index].strip()
        states4 = lines4[index].strip().split(' ')
        states5 = lines5[index].strip()

        line1 = states2[1] + ' ' + states1 + ' ' + states4[0] + ' ' + \
                states4[1] + ' ' + states3 + '\n'
        list1.append(line1)

        line2 = states2[2] + ' ' + states5 + ' '
        time1 = float(states1) * 8388608 / 20000000.0
        time2 = float(states1) * 8388608 / 50000000.0
        time3 = float(states1) * 8388608 / 100000000.0
        line2 = line2 + str(time1) + ' ' + str(time2) + ' ' + str(time3) + '\n'
        list2.append(line2)

    fo = open("data/layering/analysis/" + analysis_type + "/local.txt", "w")
    fo.writelines(list1)
    fo.close()

    fo = open("data/layering/analysis/" + analysis_type + "/server.txt", "w")
    fo.writelines(list2)
    fo.close()


def CalAvgLocalServerPeopleNum():
    list = []
    local_sum = 0
    server_sum = 0
    fo = open("data/layering/analysis/local_people_num_analysis/local_server_peoplenum.txt", "r")
    lines = fo.readlines()
    fo.close()
    for index in range(len(lines)):
        local_people_num = lines[index].strip().split(' ')[1]
        server_people_num = lines[index].strip().split(' ')[2]
        local_sum += float(local_people_num)
        server_sum += float(server_people_num)

    local_avg = round(local_sum / 100.0)
    server_avg = round(server_sum / 100.0)

    for index in range(len(lines)):
        list.append(str(index) + ' ' + str(local_avg) + ' ' + str(server_avg) + '\n')

    fo = open("data/layering/analysis/file_size_analysis/local_server_peoplenum.txt", "w")
    fo.writelines(list)
    fo.close()


def ChangeLocalServerPeopleNum():
    list = []
    fo = open("data/layering/analysis/file_size_analysis/local_server_peoplenum.txt", "w")

    for index in range(100):
        list.append(str(index) + ' ' + str(7) + ' ' + str(13) + '\n')

    fo.writelines(list)
    fo.close()


def evaluate(agent, env, type):
    obs = env.reset(type)
    evaluate_reward_list = []
    local_reward_list = []
    server_reward_list = []
    isOver = False
    while not isOver:
        action = agent.predict(obs)
        obs, evaluate_reward, local_reward, server_reward, isOver, _ = env.step(action)
        evaluate_reward_list.append(evaluate_reward)
        local_reward_list.append(local_reward)
        server_reward_list.append(server_reward)
    return evaluate_reward_list, local_reward_list, server_reward_list


def SingleStatusAnalysis(type):
    env = AnalysisEnv(type)
    act_dim = env.action_space.n
    obs_shape = env.observation_space.shape
    algo = 'DQN'

    model = AtariModel(act_dim, algo)
    if algo == 'DDQN':
        algorithm = parl.algorithms.DDQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    elif algo in ['DQN', 'Dueling']:
        algorithm = parl.algorithms.DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = AtariAgent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=act_dim)

    if algo == 'DQN':
        if os.path.exists('./dqn_models/dqn_model3'):
            agent.restore('./dqn_models/dqn_model3')
            print("DQN加载模型成功，开始预测：")

    if algo == 'DDQN':
        if os.path.exists('./ddqn_models/ddqn_model2'):
            agent.restore('./ddqn_models/ddqn_model2')
            print("DDQN加载模型成功，开始预测：")

    if algo == 'Dueling':
        if os.path.exists('./dueling_dqn_models/dueling_model2'):
            agent.restore('./dueling_dqn_models/dueling_model2')
            print("Dueling DQN加载模型成功，开始预测：")

    evaluate_reward_list, local_reward_list, server_reward_list = evaluate(agent, env, type)

    print("预测结束")

    x_np = np.arange(600)
    evaluate_np = np.asarray(evaluate_reward_list)
    local_np = np.asarray(local_reward_list)
    server_np = np.asarray(server_reward_list)

    plt.title("Analysis result")
    plt.xlabel("index")
    plt.ylabel("reward")
    plt.plot(x_np, local_np, color="red")
    plt.plot(x_np, server_np, color="blue")
    plt.plot(x_np, evaluate_np, color="black")
    plt.legend(('local', 'server', 'eval'), loc='best')
    plt.savefig(fname="data/analysis/"+type+"_analysis_result.png")
    plt.show()


if __name__ == '__main__':
    # LocalPeopleNumAnalysis()
    # AddIndex("test_local_server_peoplenum")
    SortFile()
    # CreateFile()

    # ParseTrainResult("1621150682506dqn")
    # ParseTrainResult("1621164776887ddqn")
    # ParseTrainResult("1621215782630dueling")
    # ParseTrainResult("1621232481409prioritized_dqn")

    # ParseTestResult("1621150682506dqn")
    # ParseTestResult("1621164776887ddqn")
    # ParseTestResult("1621215782630dueling")
    # ParseTestResult("1621232481409prioritized_dqn")

    # ParseTestResult("1619922206748policygradient")

    # CreateNormalDistribution("local_people_num_analysis/file_size", 0.5, 0.0001, 100)
    # CreateMemCpuNormalDistribution("local_people_num_analysis/mem_cpu_usage", 0.5, 0.0001, 100)
    # CreateNormalDistribution("local_people_num_analysis/local_protime", 1.2, 0.0001, 100)
    # CreateNormalDistribution("local_people_num_analysis/server_protime", 6.1, 0.0001, 100)
    # CreateDataset("local_people_num_analysis")

    # CalAvgLocalServerPeopleNum()
    # CreateNormalDistribution("file_size_analysis/file_size", 1, 0.5, 100)
    # CreateNormalDistribution("file_size_analysis/local_protime", 1.2, 0.00001, 100)
    # CreateNormalDistribution("file_size_analysis/server_protime", 6.1, 0.00001, 100)
    # CreateMemCpuNormalDistribution("file_size_analysis/mem_cpu_usage", 0.675, 0.00001, 100)
    # CreateDataset("file_size_analysis")

    # CreateNormalDistribution("mem_cpu_usage_analysis/file_size", 0.5, 0.0001, 100)
    # CreateMemCpuNormalDistribution("mem_cpu_usage_analysis/mem_cpu_usage", 0.5, 0.2, 100)
    # CreateNormalDistribution("mem_cpu_usage_analysis/local_protime", 1.2, 0.0001, 100)
    # CreateNormalDistribution("mem_cpu_usage_analysis/server_protime", 6.1, 0.0001, 100)
    # CreateDataset("mem_cpu_usage_analysis")

    # SingleStatusAnalysis("file_size_analysis")
    # SingleStatusAnalysis("local_people_num_analysis")
    # SingleStatusAnalysis("mem_cpu_usage_analysis")

    # ChangeLocalServerPeopleNum()
    # CreateDataset("file_size_analysis")

    # SingleStatusAnalysis("train_local")



