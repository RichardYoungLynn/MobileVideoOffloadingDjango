'''Observation:
        Type: Box(3)
        Num	Observation               Min        Max
        0	PeopleNum                 0.0         5.0
        1	BatteryRemainingLevel     0.0         1.0
        2	BandwidthQuality          0.0         100.0

Actions:
        Type: Discrete(2)
        Num	Action
        0	local
        1	offload'''

from gym import spaces
from ObjectDetection.views import getPeopleConfidenceSum, getLocalProcessTime
import numpy as np

class VideoOffloadEnv():

    state = None

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, 0.0]), np.array([20.0, 1.0, 100.0]))
        self.state = None

    def step(self, action):
        people_num, battery_remaining_level, bandwidth_quality = self.state
        if action == 0:

            reward=getPeopleConfidenceSum()/getLocalProcessTime()
        elif action == 1:
            pass

        # self.state = np.array([user_equipment_capacity, edge_server_capacity])
        #
        # done = (np.abs(user_equipment_capacity) <= 0) or (np.abs(edge_server_capacity) <= 0)
        # done = bool(done)

        if not done:
            reward = -0.1
        else:
            reward = self.detect_accuracy - self.energe_consumption + 1

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0, 0, 0])
        return self.state

    # def render(self, mode='human'):
    #     return None
    #
    # def close(self):
    #     return None

if __name__ == '__main__':
    env = VideoOffloadEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)