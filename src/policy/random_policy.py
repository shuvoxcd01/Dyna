import random
from src.envs.gridworld_mdp import GridWorld
from src.policy.base_policy import BasePolicy


class EquiprobableRandomPolicy(BasePolicy):
    def __init__(self, mdp):
        self.world_model = mdp

    def get_prob(self, selected_action, state):
        assert state in self.world_model.states
        assert selected_action in self.world_model.actions

        prob = 1.0 / float(len(self.world_model.actions))

        return prob

    def get_action(self, state):
        return random.choice(list(self.world_model.actions))
