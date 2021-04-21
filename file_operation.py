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
    result={'local_people_num':rewards[0],'local_people_confidence_sum': rewards[1],'local_process_time': rewards[2],
            'file_size': rewards[3],'memory_usage': rewards[4],'cpu_usage': rewards[5]}
    return result

def readServerReward(index,train):
    if train==1:
        line=server_lines_train[index].strip()
    elif train==0:
        line=server_lines_test[index].strip()
    rewards=line.split(' ')
    result={'server_people_num':rewards[0],'server_people_confidence_sum': rewards[1],"server_process_time":rewards[2],
            "server_transmission_time_selftest":rewards[3],"server_transmission_time_4g":rewards[4],"server_transmission_time_5g":rewards[5]}
    return result

if __name__ == '__main__':
    print(readLocalReward(0,1))
    print(readLocalReward(0,0))
    print(readServerReward(0,1))
    print(readServerReward(0,0))