import numpy as np

from gridworld import GridWorld as GW
import math


class OptimalPolicy:
    max_iterations = 1000
    discount_factor = 1.0
    epsilon = 0.1

    def __init__(self, world):
        self.world = world
        self.value_table = self.__generate_empty_matrix()

        # row, col = self.world.size
        # print([[0 for _ in range(row)] for _ in range(col)])

    def policy_iteration(self):
        pass

    def __policy_evaluation(self, policy):
        n_iterations = 0
        max_norm = 0
        for i in range(int(self.max_iterations)):
            n_iterations += 1
            max_norm = 0
            new_value_table = self.__generate_empty_matrix()

            col, row = self.world.n_states
            for x in range(row):
                for y in range(col):
                    new_value_table[x][y] = self.world.get_reward((x, y))

                    if self.world.cell_at(x, y) == GW.STATE_WHITE:
                        action = policy[x][y]
                        possibilities = self.world.step((x, y), action)
                        for a, s_t1, p in possibilities:
                            new_value_table[x][y] += p * self.value_table[s_t1[0]][s_t1[1]]
                        new_value_table[x][y] *= self.discount_factor
                    max_norm = max(max_norm, abs(self.value_table[x][y] - new_value_table[x][y]))
            self.value_table = new_value_table
            print(new_value_table)
            if max_norm <= self.epsilon * (1 - self.discount_factor) / self.discount_factor:
                break
            elif n_iterations >= self.max_iterations:
                break

    def __generate_empty_matrix(self):
        col, row = self.world.n_states
        return [[0 for _ in range(row)] for _ in range(col)]
