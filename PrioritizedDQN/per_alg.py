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

import copy

import numpy as np
import paddle.fluid as fluid

import parl
from parl.core.fluid import layers


class PrioritizedDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm with prioritized experience replay.

        Args:
            model (parl.Model): model defining forward network of Q function
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal, sample_weight):
        """ update value model self.model with DQN algorithm
        """

        pred_value = self.model.value(obs)
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True
        target = reward + (
                1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v

        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            action_onehot * pred_value, dim=1)
        delta = layers.abs(target - pred_action_value)
        cost = sample_weight * layers.square_error_cost(
            pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost, delta  # `delta` is the TD-error

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)


class PrioritizedDoubleDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ Double DQN algorithm
        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal, sample_weight):
        pred_value = self.model.value(obs)
        action_onehot = layers.one_hot(action, self.act_dim)
        pred_action_value = layers.reduce_sum(
            action_onehot * pred_value, dim=1)

        # calculate the target q value
        next_action_value = self.model.value(next_obs)
        greedy_action = layers.argmax(next_action_value, axis=-1)
        greedy_action = layers.unsqueeze(greedy_action, axes=[1])
        greedy_action_onehot = layers.one_hot(greedy_action, self.act_dim)
        next_pred_value = self.target_model.value(next_obs)
        max_v = layers.reduce_sum(
            greedy_action_onehot * next_pred_value, dim=1)
        max_v.stop_gradient = True

        target = reward + (
                1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
        delta = layers.abs(target - pred_action_value)
        cost = sample_weight * layers.square_error_cost(
            pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost, delta

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)