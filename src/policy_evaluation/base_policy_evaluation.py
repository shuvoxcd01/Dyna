from src.envs.gridworld_mdp import GridWorld


class BasePolicyEvaluation:
    def __init__(self, policy, mdp: GridWorld):
        self.states = mdp.states
        self.actions = mdp.actions
        self.reward_fn = mdp.reward_fn
        self.transition_fn = mdp.transition_fn
        self.env = mdp
        self.gamma = mdp.gamma

        self.policy = policy
        self.V = dict()

    def get_value(self, state):
        return self.V.get(state)

    def set_value(self, state, value):
        self.V[state] = value

    def get_value_fn(self):
        return lambda state: self.get_value(state)
