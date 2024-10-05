# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.algorithms.deep_cfr."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()


class DeepCFRTest(parameterized.TestCase):

  def test_deep_cfr_runs(self):
    game = pyspiel.load_game('kuhn_poker')
    with tf.Session() as sess:
      deep_cfr_solver = deep_cfr.DeepCFRSolver(
          sess,
          game,
          policy_network_layers=(8, 4),
          advantage_network_layers=(4, 2),
          num_iterations=100,
          num_traversals=10000,
          learning_rate=1e-1,
          batch_size_advantage=None,
          batch_size_strategy=None,
          policy_network_train_steps=50,
          advantage_network_train_steps=50,
          memory_capacity=1e7)
      sess.run(tf.global_variables_initializer())
      deep_cfr_solver.solve()
      tabular_policy = deep_cfr_solver.to_tabular()
      for info_state_str in tabular_policy.state_lookup.keys():
          strategies = ['{:03.2f}'.format(x)
                        for x in tabular_policy.policy_for_key(info_state_str)]
          print('{} {}'.format(info_state_str.ljust(6), strategies))
      conv = exploitability.nash_conv(
          game,
          policy.tabular_policy_from_callable(
              game, deep_cfr_solver.action_probabilities))
      print('Deep CFR NashConv: {}'.format(conv))


if __name__ == '__main__':
  tf.test.main()
