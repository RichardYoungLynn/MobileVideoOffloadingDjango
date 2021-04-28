import random
import math
import numpy as np
import os
import time
import math
import matplotlib
import matplotlib.pyplot as plt

# import parl
# from parl.utils import logger
#
# from DQN.cartpole_model import CartpoleModel
# from DQN.cartpole_agent import CartpoleAgent
# from DQN.replay_memory import ReplayMemory
# from analysis_env import AnalysisEnv

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def ParseResult(filename):
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
            train_count += 1
            train_sum += float(results[2])
            if train_count % 100 == 0:
                train_x_list.append(train_count)
                train_y_list.append(train_sum / 100.0)
                train_sum = 0
        if results[0] == 'Test':
            test_count += 1
            test_sum += float(results[2])
            if test_count % 10 == 0:
                test_x_list.append(test_count)
                test_y_list.append(test_sum / 10.0)
                test_sum = 0

    # print(len(train_x_list))
    # print(len(train_y_list))
    #
    # train_x_np = np.asarray(train_x_list)
    # train_y_np = np.asarray(train_y_list)
    #
    # print(train_x_np)
    # print(train_y_np)
    #
    # plt.title("Train result in policy gradient")
    # plt.xlabel("episode num (*100)")
    # plt.ylabel("average reward in 100 episodes")
    # plt.plot(train_x_np, train_y_np)
    # plt.savefig(fname="data/result/train_policygradient.png")
    # plt.show()

    print(len(test_x_list))
    print(len(test_y_list))

    test_x_np = np.asarray(test_x_list)
    test_y_np = np.asarray(test_y_list)

    print(test_x_np)
    print(test_y_np)

    plt.title("Test result in policy gradient")
    plt.xlabel("episode num (*10)")
    plt.ylabel("average reward in 10 episodes")
    plt.plot(test_x_np, test_y_np)
    plt.savefig(fname="data/result/test_policygradient.png")
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


def evaluate(agent, env):
    obs = env.reset(0)
    evaluate_reward = 0
    isOver = False
    while not isOver:
        action = agent.predict(obs)
        obs, reward, isOver, _ = env.step(action, 0)
        evaluate_reward += reward
    return evaluate_reward


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

    fo = open("data/layering/analysis/test_local_server_peoplenum.txt", "w")
    fo.writelines(list)
    fo.close()


def SortFile(filename):
    list = []
    fo = open("data/layering/analysis/" + filename + ".txt", "r")
    lines = fo.readlines()
    fo.close()
    len1 = len(lines)
    for index in range(len1):
        line = lines[index].strip().split(' ')
        list.append([line[0], line[1], line[2]])

    list.sort(key=TakeSecond)

    lines = []
    for index in range(len1):
        line = list[index]
        lines.append(line[0] + ' ' + line[1] + ' ' + line[2] + '\n')

    fo = open("data/layering/analysis/" + filename + ".txt", "w")
    fo.writelines(lines)
    fo.close()


# def LocalPeopleNumAnalysis():
#     env = AnalysisEnv()
#     action_dim = env.action_space.n
#     obs_shape = env.observation_space.shape
#
#     model = CartpoleModel(act_dim=action_dim)
#     algorithm = parl.algorithms.DQN(
#         model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
#     agent = CartpoleAgent(
#         algorithm,
#         obs_dim=obs_shape[0],
#         act_dim=action_dim,
#         e_greed=0.3,  # explore
#         e_greed_decrement=1e-8
#     )  # probability of exploring is decreasing during training
#
#     # 加载模型
#     if os.path.exists('./dqn_model'):
#         agent.restore('./dqn_model')
#         print("加载模型成功，开始预测：")
#
#     evaluate(agent, env)
#     action = agent.predict(obs)


# def FileSizeAnalysis():
#     list = []
#
#     for index in range(100):
#         test_local_peoplenum = lines1[index].strip().split(' ')[0]
#         test_server_peoplenum = lines2[index].strip().split(' ')[0]
#         list.append(test_local_peoplenum + ' ' + test_server_peoplenum + '\n')
#
#     fo = open("data/layering/analysis/test_local_server_peoplenum.txt", "w")
#     fo.writelines(list)
#     fo.close()


def MemCpuUsageAnalysis():
    pass


def CreateLocalPeopleNumAnalysis():



if __name__ == '__main__':
    # pass
    # LocalPeopleNumAnalysis()
    # AddIndex("test_local_server_peoplenum")
    # SortFile("test_local_server_peoplenum")
    # CreateFile()
    ParseResult("1619537109013policygradient")
