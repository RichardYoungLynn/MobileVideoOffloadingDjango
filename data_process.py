import random

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

if __name__ == '__main__':
    # createBandwidthInfo()
    # writeBandwidthInfoToDataset()
    # createLocalDataset()
    createServerDataset()