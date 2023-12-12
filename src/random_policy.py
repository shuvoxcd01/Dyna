from src.envs.gridworld_mdp import GridWorld


class EquiprobableRandomPolicy:
    def __init__(self):
        self.world_model = GridWorld()

    def get_prob(self, selected_action, state):
        assert state in self.world_model.states
        assert selected_action in self.world_model.actions

       
        prob = 1.0 / float(len(self.world_model.actions))

        return prob
    
    def get_action(self, state):
        return random.choice(list(self.world_model.actions))
