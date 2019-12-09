import numpy as np

from gridworld import GridWorld as GW
import math


class OptimalPolicy:
    max_iterations = 1000
    discount_factor = 1.0

    def __init__(self, world):
        self.world = world

        # row, col = self.world.size
        # print([[0 for _ in range(row)] for _ in range(col)])

    def policy_iteration(self):
        policy_table = np.ones([self.world.n_states, self.world.n_actions]) / self.world.n_actions
        evaluated_policies = 0
        for i in range(int(self.max_iterations)):
            stable_policy = False
            V = self.__policy_evaluation(policy_table)
            for state in range(self.world.n_states):
                current_action = np.argmax(policy_table[state])
                action_value = self.one_step_lookahead(state, V)
                best_action = np.argmax(action_value)
        print(policy_table)

    def __policy_evaluation(self, policy):
        theta = 0.001
        n_iterations = 0
        # value_table = np.zeros(self.world.size)
        # print(value_table)

        for i in range(int(self.max_iterations)):
            n_iterations += 1
            row, col = self.world.size
            value_table = [[0 for _ in range(row)] for _ in range(col)]
            for x in range(row):
                for y in range(col):
                    value_table[x][y] = self.world.get_reward((x, y))
                    if self.world.cell_at(x, y) == GW.STATE_WHITE:
                        action = policy[x][y]
                        possibilities = self.world.step((x, y), action)
                        for a, s1, p in possibilities:
                            value_table[y][x] += p * self.utilities[s1[0]][s1[1]]

            print(value_table)
            # for state in range(self.world.size):
            #    v = 0
            return 0
