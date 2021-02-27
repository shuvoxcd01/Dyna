from src.gridworld_mdp import GridWorld


class EquiprobableRandomPolicy:
    def __init__(self):
        self.world_model = GridWorld()

    def get_prob(self, selected_action, state):
        assert state in self.world_model.states
        assert selected_action in self.world_model.actions

        num_all_possible_actions = 0
        times_selected_action_chosen = 0

        for next_state in self.world_model.states:
            for action in self.world_model.actions:
                if self.world_model.reward_fn(state, action, next_state) == -1:
                    num_all_possible_actions += 1
                    if action == selected_action:
                        times_selected_action_chosen += 1

        if not num_all_possible_actions:
            return 0

        prob = times_selected_action_chosen / num_all_possible_actions

        return prob
