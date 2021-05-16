#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers



class AtariAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(AtariAgent, self).__init__(algorithm)
        self.exploration = 0.9
        self.global_step = 0
        self.update_target_steps = 200

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs',
                shape=[self.obs_dim],
                dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs',
                shape=[self.obs_dim],
                dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs',
                shape=[self.obs_dim],
                dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            sample_weight = layers.data(
                name='sample_weight', shape=[1], dtype='float32')
            self.cost, self.delta = self.alg.learn(
                obs, action, reward, next_obs, terminal, sample_weight)

    def sample(self, obs, decay_exploration=True):
        sample = np.random.rand()
        if sample < self.exploration:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)

        if decay_exploration:
            self.exploration = max(0.1, self.exploration - 1e-6)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
        return act

    def learn(self, obs, act, reward, next_obs, terminal, sample_weight):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward.astype('float32'),
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal.astype('bool'),
            'sample_weight': sample_weight.astype('float32')
        }
        cost, delta = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost, self.delta])
        return cost, delta