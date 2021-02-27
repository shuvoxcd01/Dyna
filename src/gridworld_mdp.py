from src.mdp import MDP


class GridWorld(MDP):
    def __init__(self):
        super().__init__()
        self.states = set(range(16))
        self.actions = {'up', 'down', 'right', 'left'}
        self.gamma = 1.
        self.reward_fn = self.get_reward
        self.transition_fn = self.get_transition_probability

    def get_transition_probability(self, next_state, reward, cur_state, action):
        if (reward != self.reward_fn(cur_state, action, next_state)) or (reward != -1):
            return 0

        if next_state not in self.states:
            return 0

        if cur_state not in self.states:
            return 0

        if action not in self.actions:
            return 0

        if cur_state == 0 or cur_state == 15:
            return 0

        possible_next_states = set()

        state_after_action_up = (cur_state - 4) if (cur_state - 4) in self.states else None
        if state_after_action_up is not None:
            possible_next_states.add(state_after_action_up)

        state_after_action_left = (cur_state - 1) if ((cur_state - 1) // 4 == (cur_state // 4)) else None
        if state_after_action_left is not None:
            possible_next_states.add(state_after_action_left)

        state_after_action_right = (cur_state + 1) if ((cur_state + 1) // 4 == (cur_state // 4)) else None
        if state_after_action_right is not None:
            possible_next_states.add(state_after_action_right)

        state_after_action_down = (cur_state + 4) if (cur_state + 4) in self.states else None
        if state_after_action_down is not None:
            possible_next_states.add(state_after_action_down)

        if next_state not in possible_next_states:
            return 0

        return 1

    def get_reward(self, cur_state, action, next_state):
        assert cur_state in self.states
        assert next_state in self.states
        assert action in self.actions

        if cur_state == 0 or cur_state == 15:
            return 0

        possible_next_states = dict()

        state_after_action_up = (cur_state - 4) if (cur_state - 4) in self.states else None
        if state_after_action_up is not None:
            possible_next_states['up'] = state_after_action_up

        state_after_action_left = (cur_state - 1) if ((cur_state - 1) // 4 == (cur_state // 4)) else None
        if state_after_action_left is not None:
            possible_next_states['left'] = state_after_action_left

        state_after_action_right = (cur_state + 1) if ((cur_state + 1) // 4 == (cur_state // 4)) else None
        if state_after_action_right is not None:
            possible_next_states['right'] = state_after_action_right

        state_after_action_down = (cur_state + 4) if (cur_state + 4) in self.states else None
        if state_after_action_down is not None:
            possible_next_states['down'] = state_after_action_down

        if possible_next_states.get(action) != next_state:
            return 0

        return -1
