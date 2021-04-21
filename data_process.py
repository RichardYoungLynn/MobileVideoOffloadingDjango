import random
import math

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
    fo1 = open("data/rl_data/test_local.txt", "r")
    fo2 = open("data/rl_data/test_file_size.txt", "r")
    lines1=fo1.readlines()
    lines2=fo2.readlines()
    fo1.close()
    fo2.close()
    len1=len(lines1)
    for index in range(len1):
        states=lines1[index].strip().split(' ')
        new_line=states[0]+' '+states[3]+' '+states[4]+' '+lines2[index].strip()+'\n'
        list.append(new_line)

    fo3=open("data/rl_data/test_local.txt", "w")
    fo3.writelines(list)
    fo3.close()

def createServerDataset():
    list=[]
    fo1 = open("data/rl_data/test_server.txt", "r")
    fo2 = open("data/rl_data/test_file_size.txt", "r")
    lines1=fo1.readlines()
    lines2=fo2.readlines()
    fo1.close()
    fo2.close()
    len1=len(lines1)
    for index in range(len1):
        time1=float(lines2[index].strip())/20000000.0
        time2=float(lines2[index].strip())/50000000.0
        time3=float(lines2[index].strip())/100000000.0
        new_line=lines1[index].strip()+' '+str(time1)+' '+str(time2)+' '+str(time3)+'\n'
        list.append(new_line)

    fo3=open("data/rl_data/test_server.txt", "w")
    fo3.writelines(list)
    fo3.close()

def test():
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
        people_num_sum_local+=float(states1[0])
        conf_sum_local+=float(states1[1])
        process_time_sum_local+=float(states1[2])
        people_num_sum_server+=float(states2[0])
        conf_sum_server+=float(states2[1])
        process_time_sum_server+=float(states2[2])
        transmission_time1_sum_server+=float(states2[3])
        transmission_time2_sum_server+=float(states2[4])
        transmission_time3_sum_server+=float(states2[5])

    print("people_num_avg_local="+str(people_num_sum_local/len1))
    print("conf_avg_local=" +str(conf_sum_local/len1))
    print("process_time_avg_local=" +str(process_time_sum_local/len1))
    print("people_num_avg_server=" +str(people_num_sum_server/len1))
    print("conf_avg_server=" +str(conf_sum_server/len1))
    print("process_time_avg_server=" +str(process_time_sum_server/len1))
    print("transmission_time1_avg_server=" +str(transmission_time1_sum_server/len1))
    print("transmission_time2_avg_server=" +str(transmission_time2_sum_server/len1))
    print("transmission_time3_avg_server=" +str(transmission_time3_sum_server/len1))

    print("--------------------------")
    local_reward_part1=(conf_sum_local / len1)/math.exp(process_time_sum_local/len1)
    print(str(local_reward_part1))

    server_reward_part1 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1)+(transmission_time1_sum_server/len1))
    server_reward_part2 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1) + (transmission_time2_sum_server / len1))
    server_reward_part3 = (conf_sum_server / len1) / math.exp((process_time_sum_server / len1) + (transmission_time3_sum_server / len1))
    print(str(server_reward_part1))
    print(str(server_reward_part2))
    print(str(server_reward_part3))
    print("--------------------------")
    print(math.exp(284 / 3600 *(process_time_sum_local/len1)))
    print(math.exp(541 / 3600*(process_time_sum_local/len1)))
    print(math.exp(849 / 3600*(process_time_sum_local/len1)))
    print(math.exp(1038 / 3600*(process_time_sum_local/len1)))
    print(math.exp(1209 / 3600*(process_time_sum_local/len1)))
    print(math.exp(1261 / 3600*(process_time_sum_local/len1)))
    print("--------------------------")
    print(math.exp(62 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(103 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(135 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(179 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(200 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(232 / 3600 * (process_time_sum_local / len1)))
    print(math.exp(247 / 3600 * (process_time_sum_local / len1)))


if __name__ == '__main__':
    # createBandwidthInfo()
    # writeBandwidthInfoToDataset()
    # createLocalDataset()
    # createServerDataset()
    test()
    # createMemoryCPUInfo()
    # writeMemoryCPUInfoToDataset()
