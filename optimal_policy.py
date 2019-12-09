import numpy as np

from gridworld import GridWorld as GW
import math


class OptimalPolicy:
    max_iterations = 100000
    discount_factor = 0.9
    epsilon = 0.1

    def __init__(self, world):
        self.world = world
        self.value_table = self.__generate_empty_matrix()

        # row, col = self.world.size
        # print([[0 for _ in range(row)] for _ in range(col)])

    def policy_iteration(self):
        col, row = self.world.n_states
        policy = [[(None if self.world.cell_at(x, y) == GW.STATE_WHITE else self.world.get_random_action()) for x in range(row)] for y in range(col)]

        n_iterations = 0
        run = True
        while run:
            self.__policy_evaluation(policy)
            n_iterations += 1

            changes = False
            for x in range(row):
                for y in range(col):
                    if self.world.cell_at(x, y) == GW.STATE_WHITE:
                        new_max = None
                        arg_max = None
                        for action in GW.all_actions:
                            _sum = 0
                            info = self.world.step((x, y), action)
                            for a, s_t1, p in info:
                                _sum += p * self.value_table[s_t1[0]][s_t1[1]]
                            if (new_max is None) or (_sum > new_max):
                                arg_max = action
                                new_max = _sum
                        _sum = 0
                        info = self.world.step((x, y), policy[x][y])
                        for a, s_t1, p in info:
                            _sum += p * self.value_table[s_t1[0]][s_t1[1]]
                        if new_max > _sum:
                            policy[x][y] = arg_max
                            changes = True
            run = changes
            if n_iterations >= self.max_iterations:
                run = False
                print("Ended: max iterations limit exceeded!")
        print(self.value_table)
        return n_iterations

    def __policy_evaluation(self, policy):
        n_iterations = 0
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

                        if action is None:
                            action = policy[x][y] = self.world.get_random_action()

                        info = self.world.step((x, y), action)
                        for a, s_t1, p in info:
                            new_value_table[x][y] += p * self.value_table[s_t1[0]][s_t1[1]]
                        new_value_table[x][y] *= self.discount_factor
                    max_norm = max(max_norm, abs(self.value_table[x][y] - new_value_table[x][y]))
            self.value_table = new_value_table
            if max_norm <= self.epsilon * (1 - self.discount_factor) / self.discount_factor:
                print("Policy evaluated in {} iterations.".format(n_iterations))
                break
            elif n_iterations >= self.max_iterations:
                print("Ended: max iterations limit exceeded!--")
                break

    def __generate_empty_matrix(self):
        col, row = self.world.n_states
        return [[0 for _ in range(row)] for _ in range(col)]
