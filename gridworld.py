import random


class GridWorld:
    STATE_WHITE = 0  # can walk
    STATE_RED = 1  # danger
    STATE_GREEN = 2  # goal
    STATE_BLACK = 3  # wall
    # STATE_YELLOW = 3  # start

    ACTION_EAST = 'East'
    ACTION_WEST = 'West'
    ACTION_NORTH = 'North'
    ACTION_SOUTH = 'South'

    PROB_FORWARD = 'Forward'
    PROB_BACKWARD = 'Backward'
    PROB_LEFT = 'Left'
    PROB_RIGHT = 'Right'

    all_actions = (ACTION_NORTH, ACTION_SOUTH, ACTION_WEST, ACTION_EAST)

    def __init__(self, _map):
        self.__map = _map
        self.__prob = {self.PROB_FORWARD: 1,
                       self.PROB_BACKWARD: 0,
                       self.PROB_LEFT: 0,
                       self.PROB_RIGHT: 0}
        self.__reward = {self.STATE_WHITE: 0,
                         self.STATE_RED: -1,
                         self.STATE_GREEN: 1,
                         self.STATE_BLACK: 0}
        self.n_states = len(self.__map) * len(self.__map[0])
        self.n_actions = len(self.all_actions)

    def set_prob(self, forward, backward, left, right):
        if forward + backward + left + right != 1:
            raise Exception('Probabilities are incorrect, sum is not 1!')
        self.__prob = {self.PROB_FORWARD: forward,
                       self.PROB_BACKWARD: backward,
                       self.PROB_LEFT: left,
                       self.PROB_RIGHT: right}

    def set_reward(self, state_white, state_red, state_green):
        self.__reward = {self.STATE_WHITE: state_white,
                         self.STATE_RED: state_red,
                         self.STATE_GREEN: state_green,
                         self.STATE_BLACK: 0}

    def get_reward(self, state):
        return self.__reward[self.__map[state[0]][state[1]]]

    def get_random_action(self):
        return self.all_actions[int(random.random() * 4)]

    def cell_at(self, x, y):
        return self.__map[x][y]

    def step(self, state, action):
        if self.__map[state[0]][state[1]] != self.STATE_WHITE:
            raise Exception('An action is not allowed here!')

        prob = self.__get_relative_prob(action)
        states = []
        for action in self.all_actions:
            states.append((action, self.__move(state, action), prob[action]))
        return states

    def __move(self, state, action):
        new_state = state
        if action == self.ACTION_EAST:
            new_state = (state[0], min(len(self.__map[0]) - 1, state[1] + 1))
        elif action == self.ACTION_WEST:
            new_state = (state[0], max(0, state[1] - 1))
        elif action == self.ACTION_NORTH:
            new_state = (max(0, state[0] - 1), state[1])
        elif action == self.ACTION_SOUTH:
            new_state = (min(len(self.__map) - 1, state[0] + 1), state[1])

        if self.__map[new_state[0]][new_state[1]] == self.STATE_BLACK:
            return state
        else:
            return new_state

    def __get_relative_prob(self, action):
        if action == self.ACTION_EAST:
            return {self.ACTION_NORTH: self.__prob[self.PROB_LEFT],
                    self.ACTION_SOUTH: self.__prob[self.PROB_RIGHT],
                    self.ACTION_WEST: self.__prob[self.PROB_BACKWARD],
                    self.ACTION_EAST: self.__prob[self.PROB_FORWARD]}
        elif action == self.ACTION_WEST:
            return {self.ACTION_NORTH: self.__prob[self.PROB_RIGHT],
                    self.ACTION_SOUTH: self.__prob[self.PROB_LEFT],
                    self.ACTION_WEST: self.__prob[self.PROB_FORWARD],
                    self.ACTION_EAST: self.__prob[self.PROB_BACKWARD]}
        elif action == self.ACTION_NORTH:
            return {self.ACTION_NORTH: self.__prob[self.PROB_FORWARD],
                    self.ACTION_SOUTH: self.__prob[self.PROB_BACKWARD],
                    self.ACTION_WEST: self.__prob[self.PROB_LEFT],
                    self.ACTION_EAST: self.__prob[self.PROB_RIGHT]}
        elif action == self.ACTION_SOUTH:
            return {self.ACTION_NORTH: self.__prob[self.PROB_BACKWARD],
                    self.ACTION_SOUTH: self.__prob[self.PROB_FORWARD],
                    self.ACTION_WEST: self.__prob[self.PROB_RIGHT],
                    self.ACTION_EAST: self.__prob[self.PROB_LEFT]}
