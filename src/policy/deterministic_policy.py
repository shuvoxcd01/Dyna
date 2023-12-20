from src.policy.base_policy import BasePolicy
from typing import Dict, Optional


class DeterministicPolicy(BasePolicy):
    def __init__(self, state_action_map: Optional[Dict] = None):
        super().__init__()
        self.state_action_map = state_action_map if state_action_map is not None else {}

    def get_prob(self, selected_action, state):
        action = self.state_action_map.get(state, None)

        if action == selected_action:
            return 1.0

        return 0.0

    def get_action(self, state):
        return self.state_action_map.get(state, None)

    def update_policy(self, state, new_action):
        self.state_action_map[state] = new_action
