import gym
from QLearning.agent import QLearningAgent
from video_offload import VideoOffloadEnv
import time


def run_episode(env, agent):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset(1)  # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action, 1)  # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset(0)
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action, 0)
        total_reward += reward
        obs = next_obs
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    env = VideoOffloadEnv()
    act_n = env.action_space.n
    obs_n = env.observation_space.shape[0]

    agent = QLearningAgent(
        obs_n=obs_n,
        act_n=act_n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    max_episode = 20

    # start train
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(0, 5):
            ep_reward, ep_steps = run_episode(env, agent)
            print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

        test_episode(env, agent)


if __name__ == "__main__":
    main()