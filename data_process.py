import random
import math
import numpy as np
import matplotlib.pyplot as plt

def createBandwidthInfo():
    num = 232
    list = []
    fo = open("data/rl_data/bandwidth_info.txt", "w")

    for index in range(num):
        list.append("0\n")
        list.append("1\n")
        list.append("2\n")
        list.append("3\n")

    random.shuffle(list)
    fo.writelines(list)
    fo.close()


def createMemoryCPUInfo():
    num = 300
    list = []
    fo = open("data/rl_data/test_memory_cpu_info.txt", "w")

    for index in range(num):
        list.append(str(random.random())+" "+str(random.random())+"\n")

    fo.writelines(list)
    fo.close()


def writeMemoryCPUInfoToDataset():
    list=[]
    fo1=open("data/rl_data/test_memory_cpu_info.txt", "r")
    fo2=open("data/rl_data/test_local.txt", "r")
    lines1=fo1.readlines()
    lines2=fo2.readlines()
    fo1.close()
    fo2.close()
    # print(len(lines2))
    for index in range(len(lines2)):
        line=lines2[index].strip()+" "+lines1[index].strip()+"\n"
        # print(line)
        list.append(line)

    fo3 = open("data/rl_data/test_local.txt", "w")
    fo3.writelines(list)
    fo3.close()


def writeBandwidthInfoToDataset():
    list=[]
    fo1=open("data/rl_data/bandwidth_info.txt", "r")
    fo2=open("data/rl_data/train_local.txt", "r")
    lines1=fo1.readlines()
    lines2=fo2.readlines()
    fo1.close()
    fo2.close()
    print(len(lines2))
    for index in range(len(lines2)):
        line=lines2[index].strip()+" "+lines1[index].strip()+"\n"
        # print(line)
        list.append(line)

    fo3 = open("data/rl_data/train_local.txt", "w")
    fo3.writelines(list)
    fo3.close()


def createLocalDataset():
    list=[]
    fo1 = open("data/rl_data/test_file_size.txt", "r")
    fo2 = open("data/rl_data/test_local.txt", "r")
    fo3 = open("data/rl_data/test_local_server_process_time.txt", "r")
    fo4 = open("data/rl_data/test_mem_cpu_usage.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    lines3 = fo3.readlines()
    lines4 = fo4.readlines()
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()
    random.shuffle(lines1)
    random.shuffle(lines3)
    random.shuffle(lines4)
    len1=len(lines1)
    for index in range(len1):
        states1 = lines1[index].strip().split(' ')
        states2 = lines2[index].strip().split(' ')
        states3 = lines3[index].strip().split(' ')
        states4 = lines4[index].strip().split(' ')
        line=states2[0]+' '+states2[1]+' '+states1[1]+' '+states4[1]+' '+states4[2]+' '+states2[2]+' '+states3[1]+"\n"
        list.append(line)

    fo=open("data/rl_data/test_local.txt", "w")
    fo.writelines(list)
    fo.close()


def createServerDataset():
    list = []
    # fo1 = open("data/rl_data/test_server.txt", "r")
    # fo2 = open("data/rl_data/train_local_server_process_time.txt", "r")
    # fo3 = open("data/rl_data/train_file_size.txt", "r")
    # lines1 = fo1.readlines()
    # lines2 = fo2.readlines()
    # lines3 = fo3.readlines()
    # fo1.close()
    # fo2.close()
    # fo3.close()
    # random.shuffle(lines1)
    # random.shuffle(lines3)
    # random.shuffle(lines4)
    # len1 = len(lines1)
    # for index in range(len1):
    #     states1 = lines1[index].strip().split(' ')
    #     states2 = lines2[index].strip().split(' ')
    #     states3 = lines3[index].strip().split(' ')
    #     states4 = lines4[index].strip().split(' ')
    #     line = states2[0] + ' ' + states2[1] + ' ' + states1[1] + ' ' + states4[1] + ' ' + states4[2] + ' ' + states2[
    #         2] + ' ' + states3[1] + "\n"
    #     list.append(line)
    #
    # fo = open("data/rl_data/test_local.txt", "w")
    # fo.writelines(list)
    # fo.close()


def createDataset():
    list1 = []
    list2 = []
    fo1 = open("data/rl_data/train_file_size.txt", "r")
    fo2 = open("data/rl_data/train_local.txt", "r")
    fo3 = open("data/rl_data/train_local_server_process_time.txt", "r")
    fo4 = open("data/rl_data/train_mem_cpu_usage.txt", "r")
    fo5 = open("data/rl_data/train_server.txt", "r")
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
    random.shuffle(lines1)
    random.shuffle(lines3)
    random.shuffle(lines4)
    len1 = len(lines1)
    for index in range(len1):
        states1 = lines1[index].strip().split(' ')
        states2 = lines2[index].strip().split(' ')
        states3 = lines3[index].strip().split(' ')
        states4 = lines4[index].strip().split(' ')
        states5 = lines5[index].strip().split(' ')

        line1 = states2[0] + ' ' + states2[1] + ' ' + states1[1] + ' ' + states4[1] + ' ' + states4[2] + ' ' + states2[2] + ' ' + states3[1] + '\n'
        list1.append(line1)

        line2 = states5[0] + ' ' + states5[1] + ' ' + states5[2] + ' ' + states3[2] + ' '
        time1 = float(states1[1]) / 20000000.0
        time2 = float(states1[1]) / 50000000.0
        time3 = float(states1[1]) / 100000000.0
        line2 = line2 + str(time1) + ' ' + str(time2) + ' ' + str(time3) + '\n'
        list2.append(line2)

    fo = open("data/rl_data/train_local.txt", "w")
    fo.writelines(list1)
    fo.close()

    fo = open("data/rl_data/train_server.txt", "w")
    fo.writelines(list2)
    fo.close()


def test():
    file_size_sum_local=0.0
    people_num_sum_local=0.0
    people_num_sum_server=0.0
    conf_sum_local=0.0
    conf_sum_server=0.0
    process_time_sum_local=0.0
    process_time_sum_server=0.0
    transmission_time1_sum_server=0.0
    transmission_time2_sum_server = 0.0
    transmission_time3_sum_server = 0.0
    fo1 = open("data/rl_data/train_local.txt", "r")
    fo2 = open("data/rl_data/train_server.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    fo1.close()
    fo2.close()
    len1 = len(lines1)
    for index in range(len1):
        states1 = lines1[index].strip().split(' ')
        states2 = lines2[index].strip().split(' ')
        people_num_sum_local+=float(states1[1])
        conf_sum_local+=float(states1[5])
        process_time_sum_local+=float(states1[6])
        file_size_sum_local+=float(states1[2])
        people_num_sum_server+=float(states2[1])
        conf_sum_server+=float(states2[2])
        process_time_sum_server+=float(states2[3])
        transmission_time1_sum_server+=float(states2[4])
        transmission_time2_sum_server+=float(states2[5])
        transmission_time3_sum_server+=float(states2[6])

    print("people_num_avg_local="+str(people_num_sum_local/len1))
    print("conf_avg_local=" +str(conf_sum_local/len1))
    print("process_time_avg_local=" +str(process_time_sum_local/len1))
    print("file_size_avg_local=" + str(file_size_sum_local / len1))
    print("people_num_avg_server=" +str(people_num_sum_server/len1))
    print("conf_avg_server=" +str(conf_sum_server/len1))
    print("process_time_avg_server=" +str(process_time_sum_server/len1))
    print("transmission_time1_avg_server=" +str(transmission_time1_sum_server/len1))
    print("transmission_time2_avg_server=" +str(transmission_time2_sum_server/len1))
    print("transmission_time3_avg_server=" +str(transmission_time3_sum_server/len1))

    print("--------------------------")
    local_reward_part1=(conf_sum_local / len1)/(process_time_sum_local/len1)
    print("local_reward_part1="+str(local_reward_part1))

    server_reward_time1 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1)+(transmission_time1_sum_server/len1))
    server_reward_time2 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1) + (transmission_time2_sum_server / len1))
    server_reward_time3 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1) + (transmission_time3_sum_server / len1))
    print("server_reward_time1="+str(server_reward_time1))
    print("server_reward_time2="+str(server_reward_time2))
    print("server_reward_time3="+str(server_reward_time3))
    print("--------------------------")
    # print(math.exp(284 / 3600 *(process_time_sum_local/len1)))
    # print(math.exp(541 / 3600*(process_time_sum_local/len1)))
    # print(math.exp(849 / 3600*(process_time_sum_local/len1)))
    # print(math.exp(1038 / 3600*(process_time_sum_local/len1)))
    # print(math.exp(1209 / 3600*(process_time_sum_local/len1)))
    # print(math.exp(1261 / 3600*(process_time_sum_local/len1)))
    # print("--------------------------")
    # print(math.exp(62 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(103 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(135 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(179 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(200 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(232 / 3600 * (process_time_sum_local / len1)))
    # print(math.exp(247 / 3600 * (process_time_sum_local / len1)))


def createLocalPeopleNumNormalDistribution():
    list=[]
    fo = open("data/rl_data/local_people_num.txt", "w")
    mu, sigma = 10, 4  # 均值和标准差
    s = np.random.normal(mu, sigma, 910)
    s.sort()
    for index in range(len(s)):
        if s[index]>=0 and s[index]<=10:
            list.append(str(round(s[index]))+"\n")

    fo.writelines(list)
    fo.close()
    # count, bins, ignored = plt.hist(s, 30, density=True, color='b')
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
    #          color='r')
    # plt.show()


def createFileSizeNormalDistribution():
    list = []
    fo = open("data/rl_data/test_file_size.txt", "w")
    mu, sigma = 5000000, 200000  # 均值和标准差
    s = np.random.normal(mu, sigma, 321)
    s.sort()
    for index in range(len(s)):
        list.append(str(round(s[index])) + "\n")

    fo.writelines(list)
    fo.close()
    # count, bins, ignored = plt.hist(s, 30, density=True, color='b')
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
    #          color='r')
    # plt.show()


def createMemCpuUsageNormalDistribution():
    list = []
    fo = open("data/rl_data/test_mem_cpu_usage.txt", "w")
    mu, sigma = 0.5, 0.1  # 均值和标准差
    s1 = np.random.normal(mu, sigma, 321)
    s2 = np.random.normal(mu, sigma, 321)
    s1.sort()
    s2.sort()
    for index in range(len(s1)):
        list.append(str(s1[index]) + " " + str(s2[index]) + "\n")

    fo.writelines(list)
    fo.close()
    # count, bins, ignored = plt.hist(s1, 30, density=True, color='b')
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
    #          color='r')
    # plt.show()


def createLocalAndServerProcessTimeNormalDistribution():
    list = []
    fo = open("data/rl_data/test_local_server_process_time.txt", "w")
    mu1, sigma1 = 1.2, 0.07
    mu2, sigma2 = 6.1, 0.07
    s1 = np.random.normal(mu1, sigma1, 321)
    s2 = np.random.normal(mu2, sigma2, 321)
    s1.sort()
    s2.sort()
    for index in range(len(s1)):
        list.append(str(s1[index]) + " " + str(s2[index]) + "\n")

    fo.writelines(list)
    fo.close()

    # count, bins, ignored = plt.hist(s1, 30, density=True, color='b')
    # plt.plot(bins, 1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu1) ** 2 / (2 * sigma1 ** 2)), linewidth=2,
    #          color='r')
    # plt.show()
    #
    # count, bins, ignored = plt.hist(s2, 30, density=True, color='b')
    # plt.plot(bins, 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu2) ** 2 / (2 * sigma2 ** 2)), linewidth=2,
    #          color='r')
    # plt.show()


def addIndex(filename):
    list = []
    fo = open("data/rl_data/"+filename+".txt", "r")
    lines = fo.readlines()
    fo.close()
    for index in range(len(lines)):
        line = str(index) + " " + lines[index].strip() + "\n"
        list.append(line)

    fo = open("data/rl_data/"+filename+".txt", "w")
    fo.writelines(list)
    fo.close()


def shuffleDataset():
    fo1 = open("data/layering/train/train_local2.txt", "r")
    fo2 = open("data/layering/train/train_server2.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    fo1.close()
    fo2.close()

    state = np.random.get_state()
    np.random.shuffle(lines1)

    np.random.set_state(state)
    np.random.shuffle(lines2)

    fo = open("data/layering/train/train_local3.txt", "w")
    fo.writelines(lines1)
    fo.close()
    fo = open("data/layering/train/train_server3.txt", "w")
    fo.writelines(lines2)
    fo.close()


def createTestEnv():
    num = 232
    list = []
    fo = open("data/rl_data/train_local.txt", "w")

    for index in range(num):
        list.append("0\n")
        list.append("1\n")
        list.append("2\n")
        list.append("3\n")

    random.shuffle(list)
    fo.writelines(list)
    fo.close()


def layering():
    pass


if __name__ == '__main__':
    # createBandwidthInfo()
    # writeBandwidthInfoToDataset()
    # createLocalDataset()
    # createServerDataset()
    # test()
    # createMemoryCPUInfo()
    # writeMemoryCPUInfoToDataset()
    # createLocalPeopleNumNormalDistribution()
    # createFileSizeNormalDistribution()
    # createMemCpuUsageNormalDistribution()
    # createLocalAndServerProcessTimeNormalDistribution()

    # addIndex("test_file_size")
    # addIndex("test_local")
    # addIndex("test_local_server_process_time")
    # addIndex("test_mem_cpu_usage")
    # addIndex("test_server")

    # addIndex("train_file_size")
    # addIndex("train_local")
    # addIndex("train_local_server_process_time")
    # addIndex("train_mem_cpu_usage")
    # addIndex("train_server")

    # createDataset()

    shuffleDataset()
    # a = [0,1,2,3,4]
    # b = [10,11,12,13,14]
    # print(a, b)
    # # result:[0 1 2 3 4 5 6 7 8 9] [10 11 12 13 14 15 16 17 18 19]
    # state = np.random.get_state()
    # np.random.shuffle(a)
    # print(a)
    # # result:[6 4 5 3 7 2 0 1 8 9]
    # np.random.set_state(state)
    # np.random.shuffle(b)
    # print(b)
    # result:[16 14 15 13 17 12 10 11 18 19]
