from src.envs.gridworld_mdp import GridWorld
import numpy as np
from src.policy_evaluation.base_policy_evaluation import BasePolicyEvaluation
from src.utils.vis_util import print_grid

class ExactPolicyEvaluation(BasePolicyEvaluation):
    def __init__(self, policy, mdp: GridWorld):
        super().__init__(policy, mdp)

    def estimate_state_value_function_with_system_of_linear_equations(
        self, theta=0.001
    ):
        num_states = self.env.nS

        coeff_mat = np.zeros(shape=(self.env.nS, self.env.nS), dtype=np.float32)
        reward_mat = np.zeros(shape=(self.env.nS), dtype=np.float32)

        coeff_mat[0, 0] = 1.0
        coeff_mat[self.env.nS - 1, self.env.nS - 1] = 1.0

        for i in range(1, self.env.nS - 1):
            coeff_mat[i, i] = 1.0
            r = 0.0

            for action in self.actions:
                state = self.env.states[i]
                action_prob = self.policy.get_prob(action, state)

                transition_prob, next_state, reward, done = self.env.P[state][action][0]
                r += reward * transition_prob * action_prob

                next_state_index = self.env.states.index(next_state)

                coeff_mat[i][next_state_index] = (
                    coeff_mat[i][next_state_index]
                    - self.gamma * transition_prob * action_prob
                )

            reward_mat[i] = r

        state_values = np.linalg.solve(coeff_mat, reward_mat)

        for i in range(self.env.nS):
            self.set_value(self.env.states[i], state_values[i])

        print_grid(value_fn=self.get_value_fn(), states=self.states)

        return state_values
