import random
import math
import numpy as np
import matplotlib.pyplot as plt

def SortFile(filename):
    fo = open("data/rl_data/"+filename+".txt", "r")
    lines=fo.readlines()
    for i in range(len(lines)):
        lines[i].strip()
    fo.close()


def CreateNormalDistribution(filename,mu,sigma,num):
    list = []
    fo = open("data/layering/"+filename+".txt", "w")
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


def CreateMemCpuNormalDistribution(filename,mu,sigma,num):
    list = []
    fo = open("data/layering/"+filename+".txt", "w")
    s1 = np.random.normal(mu, sigma, num)
    s2 = np.random.normal(mu, sigma, num)
    s1.sort()
    s2.sort()
    for index in range(len(s1)):
        list.append(str(s1[index]) + " " + str(s2[index]) + "\n")

    fo.writelines(list)
    fo.close()


def AddIndex(filename):
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


def CreatePeopleNumAndConfSum(source,destination):
    list=[]
    fo = open("data/layering/" + source + ".txt", "r")
    lines = fo.readlines()
    fo.close()
    for i in range(200):
        list.append(lines[i].strip()+"\n")
    fo = open("data/layering/" + destination + ".txt", "w")
    fo.writelines(list)
    fo.close()


def CreateDataset():
    list1 = []
    list2 = []
    fo1 = open("data/layering/train_file_size.txt", "r")
    fo2 = open("data/layering/train_local_peoplenum_confsum.txt", "r")
    fo3 = open("data/layering/train_local_protime.txt", "r")
    fo4 = open("data/layering/train_mem_cpu_usage.txt", "r")
    fo5 = open("data/layering/train_server_peoplenum_confsum.txt", "r")
    fo6 = open("data/layering/train_server_protime.txt", "r")
    lines1 = fo1.readlines()
    lines2 = fo2.readlines()
    lines3 = fo3.readlines()
    lines4 = fo4.readlines()
    lines5 = fo5.readlines()
    lines6 = fo6.readlines()
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()
    fo5.close()
    fo6.close()
    random.shuffle(lines1)
    random.shuffle(lines3)
    random.shuffle(lines4)
    random.shuffle(lines6)
    len1 = len(lines1)
    for index in range(len1):
        states1 = lines1[index].strip()
        states2 = lines2[index].strip().split(' ')
        states3 = lines3[index].strip()
        states4 = lines4[index].strip().split(' ')
        states5 = lines5[index].strip().split(' ')
        states6 = lines6[index].strip()

        # print(states2[0])
        # print(states1)
        # print(states4[0])
        # print(states4[1])
        # print(states2[1])
        # print(states3)

        line1 = states2[0] + ' ' + states1 + ' ' + states4[0] + ' ' + states4[1] + ' ' + states2[1] + ' ' + states3 + '\n'
        list1.append(line1)

        line2 = states5[0] + ' ' + states5[1] + ' ' + states6 + ' '
        time1 = float(states1) * 8388608 / 20000000.0
        time2 = float(states1) * 8388608 / 50000000.0
        time3 = float(states1) * 8388608 / 100000000.0
        line2 = line2 + str(time1) + ' ' + str(time2) + ' ' + str(time3) + '\n'
        list2.append(line2)

    fo = open("data/layering/train_local.txt", "w")
    fo.writelines(list1)
    fo.close()

    fo = open("data/layering/train_server.txt", "w")
    fo.writelines(list2)
    fo.close()


if __name__ == '__main__':
    # SortFile("train_file_size")

    # CreateNormalDistribution("train_file_size",0.5,0.1,200)
    # CreateMemCpuNormalDistribution("train_mem_cpu_usage", 0.5, 0.1, 200)
    # CreateNormalDistribution("train_mem_cpu_usage", 0.5, 0.1, 200)
    # CreateNormalDistribution("train_local_protime", 1.2, 0.1, 200)
    # CreateNormalDistribution("train_server_protime", 6.1, 0.1, 200)

    # CreatePeopleNumAndConfSum("train_local","train_local_peoplenum_confsum")
    # CreatePeopleNumAndConfSum("train_server", "train_server_peoplenum_confsum")

    CreateDataset()