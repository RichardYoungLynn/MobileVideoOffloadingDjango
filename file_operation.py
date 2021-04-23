import math

local_lines_train=open("data/rl_data/train_local.txt","r").readlines()
server_lines_train=open("data/rl_data/train_server.txt","r").readlines()

local_lines_test=open("data/rl_data/test_local.txt","r").readlines()
server_lines_test=open("data/rl_data/test_server.txt","r").readlines()

def readLocalReward(index,train):
    if train==1:
        line = local_lines_train[index].strip()
    elif train==0:
        line = local_lines_test[index].strip()
    rewards=line.split(' ')
    result={'index':rewards[0],'local_people_num':rewards[1],'file_size': rewards[2],'memory_usage': rewards[3],
            'cpu_usage': rewards[4],'local_confidence_sum': rewards[5],'local_process_time': rewards[6]}
    return result


def readServerReward(index,train):
    if train==1:
        line=server_lines_train[index].strip()
    elif train==0:
        line=server_lines_test[index].strip()
    rewards=line.split(' ')
    result={'index':rewards[0],'server_people_num':rewards[1],'server_confidence_sum': rewards[2],"server_process_time":rewards[3],
            "server_transmission_time_selftest":rewards[4],"server_transmission_time_4g":rewards[5],"server_transmission_time_5g":rewards[6]}
    return result


def test():
    train=0
    train_len=930
    test_len=320
    local=0
    server=0
    max_reward=0.0
    for i in range(test_len):
        local_confidence_sum = float(readLocalReward(i, train)['local_confidence_sum'])
        local_process_time = float(readLocalReward(i, train)['local_process_time'])
        memory_usage = float(readLocalReward(i, train)['memory_usage'])
        cpu_usage = float(readLocalReward(i, train)['cpu_usage'])
        local_r1 = local_confidence_sum / local_process_time
        local_r2 = math.exp((memory_usage + cpu_usage) * 300 * local_process_time * 0.001)
        local_reward = local_r1 - local_r2

        server_confidence_sum = float(readServerReward(i, train)['server_confidence_sum'])
        server_process_time = float(readServerReward(i, train)['server_process_time'])
        server_transmission_time_selftest = float(readServerReward(i, train)['server_transmission_time_selftest'])
        server_transmission_time_4g = float(readServerReward(i, train)['server_transmission_time_4g'])
        server_transmission_time_5g = float(readServerReward(i, train)['server_transmission_time_5g'])
        server_reward = server_confidence_sum / (server_process_time + server_transmission_time_5g)

        offload = local_reward<server_reward
        if offload:
            server+=1
            max_reward+=server_reward
        else:
            local+=1
            max_reward += local_reward
        print("index = "+str(i)+", local_reward = "+str(local_reward)+", server_reward = "+str(server_reward)
              +", local_r1 = "+str(local_r1)+", local_r2 = "+str(local_r2)+", offload = "+str(local_reward<server_reward))

    print("local = "+str(local)+", server = "+str(server)+", max_reward = "+str(max_reward))


if __name__ == '__main__':
    # print(readLocalReward(0,1))
    # print(readLocalReward(0,0))
    # print(readServerReward(0,1))
    # print(readServerReward(0,0))
    test()