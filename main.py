from gridworld import GridWorld as GW
from optimal_policy import OptimalPolicy as OP


def main():
    grid_world = GW(
        [[GW.STATE_RED, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_RED, GW.STATE_BLACK, GW.STATE_BLACK, GW.STATE_GREEN],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_BLACK, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_RED, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_RED, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_BLACK, GW.STATE_WHITE],
         [GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_BLACK, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE, GW.STATE_WHITE]]
    )
    grid_world.set_prob(0.7, 0, 0.15, 0.15)  # forward, backward, left, right
    grid_world.set_reward(0.02, -1, 1)  # state_white, state_red, state_green

    print(grid_world.step((1, 0), GW.ACTION_EAST))
    print(grid_world.step((3, 3), GW.ACTION_WEST))
    print(grid_world.step((3, 3), GW.ACTION_NORTH))
    print(grid_world.step((3, 3), GW.ACTION_SOUTH))
    print(grid_world.get_reward((0, 0)))

    policy = OP(grid_world)
    policy.policy_iteration()


if __name__ == "__main__":
    main()
