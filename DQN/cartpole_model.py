#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import parl
from parl import layers


class CartpoleModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        hid3_size = 128
        hid4_size = 128
        hid5_size = 128
        hid6_size = 128
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='relu')
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=hid5_size, act='relu')
        self.fc6 = layers.fc(size=hid6_size, act='relu')
        self.fc7 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h4 = self.fc4(h3)
        h5 = self.fc5(h4)
        h6 = self.fc6(h5)
        Q = self.fc7(h6)
        return Q