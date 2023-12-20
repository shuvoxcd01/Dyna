from src.envs.gridworld import UP, DOWN, LEFT, RIGHT
from src.policy.base_policy import BasePolicy

ACTIONS = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


def print_value_grid(value_fn, states, k=None):
    states = list(states)

    if k is not None:
        print(f"k={k}")

    for i in range(len(states)):
        if not i % 4:
            print()
            print("----------------------------------")
            print("|{:6.1f} |".format(value_fn(states[i])), end="")
        else:
            print("{:6.1f} |".format(value_fn(states[i])), end="")

    print()
    print("----------------------------------")


def print_policy_grid(policy: BasePolicy, states, k=None):
    states = list(states)

    if k is not None:
        print(f"k={k}")

    for i in range(len(states)):
        if not i % 4:
            print()
            print("----------------------------------")
            print("|{:6s} |".format(ACTIONS[policy.get_action(states[i])]), end="")
        else:
            print("{:6s} |".format(ACTIONS[policy.get_action(states[i])]), end="")

    print()
    print("----------------------------------")
