from copy import deepcopy
from src.envs.gridworld_mdp import GridWorld
from src.utils.vis_util import print_grid
from src.base_policy_evaluation import BasePolicyEvaluation


class IterativePolicyEvaluation(BasePolicyEvaluation):
    def __init__(self, policy, mdp: GridWorld):
        super().__init__(policy, mdp)

    def estimate_state_value_function_inplace(self, theta=0.001, print_each_iter=False):
        """
        :param theta: A small threshold theta > 0 - determining accuracy of estimation.
        :return: An estimated state-value function (as a python dictionary).
        """
        for state in self.states:
            self.set_value(state, 0)

        k = 0
        delta = theta + 1

        while not delta < theta:
            delta = 0
            k += 1

            for state in self.states:
                v = self.get_value(state)
                self.set_value(state, self.get_updated_state_value(state))

                delta = max(delta, abs(v - self.get_value(state)))
            if print_each_iter:
                print_grid(value_fn=self.get_value_fn(), states=self.states, k=k)

        print_grid(value_fn=self.get_value_fn(), states=self.states, k=k)

    def get_updated_state_value(self, cur_state):
        updated_state_value = 0
        for action in self.actions:
            action_prob = self.policy.get_prob(action, cur_state)
            state_value_after_taking_action = 0
            for next_state in self.states:
                reward = self.reward_fn(cur_state, action)
                transition_prob = self.transition_fn(
                    next_state, reward, cur_state, action
                )
                state_value_after_taking_action += transition_prob * (
                    reward + self.gamma * self.get_value(next_state)
                )

            updated_state_value += action_prob * state_value_after_taking_action
        return updated_state_value

    def estimate_state_value_function(self, theta=0.001, print_each_iter=False):
        """
        :param theta: A small threshold theta > 0 - determining accuracy of estimation.
        :return: An estimated state-value function (as a python dictionary).
        """
        for state in self.states:
            self.set_value(state, 0)

        k = 0
        delta = theta + 1

        while not delta < theta:
            delta = 0
            k += 1

            new_values = {}

            for state in self.states:
                v = self.get_value(state)
                new_values[state] = self.get_updated_state_value(state)

                delta = max(delta, abs(v - new_values[state]))

            self.V = new_values
            if print_each_iter:
                print_grid(value_fn=self.get_value_fn(), states=self.states, k=k)

        print_grid(value_fn=self.get_value_fn(), states=self.states, k=k)
