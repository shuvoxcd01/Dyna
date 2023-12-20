import random
from typing import Optional
from src.envs.gridworld_mdp import GridWorld
from src.policy.base_policy import BasePolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.random_policy import EquiprobableRandomPolicy
from src.policy_evaluation.base_policy_evaluation import BasePolicyEvaluation
from src.policy_evaluation.exact_policy_evaluation import ExactPolicyEvaluation
from src.policy_evaluation.iterative_policy_evaluation import IterativePolicyEvaluation


class PolicyIteration:
    def __init__(self, mdp: GridWorld, policy_evaluator: BasePolicyEvaluation) -> None:
        self.mdp = mdp
        self.policy_evaluator = policy_evaluator

    def _initialize_policy(self):
        assert self.mdp.gamma != 1.0 , "Policy evaluation may not converge for gamma = 1.0"

        states = self.mdp.states
        actions = self.mdp.actions

        state_action_map = {}

        for state in states:
            state_action_map[state] = random.choice(actions)

        policy = DeterministicPolicy(state_action_map=state_action_map)

        return policy

    def do_policy_iteration(self, initial_policy: Optional[BasePolicy] = None):
        old_policy = initial_policy
        if old_policy is None:
            old_policy = self._initialize_policy()

        while True:
            V = self.policy_evaluator.evaluate_policy(old_policy)

            policy = DeterministicPolicy()

            is_policy_stable = True

            for state in self.mdp.states:
                old_action = old_policy.get_action(state)
                actions = self.mdp.actions
                action_values = {}

                for action in actions:
                    prob, _next_state, reward, done = self.mdp.P[state][action][0]
                    next_state_value = V[_next_state]
                    action_values[action] = prob * (
                        reward + self.mdp.gamma * next_state_value
                    )

                new_action = max(action_values, key=action_values.get)
                policy.update_policy(state=state, new_action=new_action)

                if old_action != new_action:
                    is_policy_stable = False

            if is_policy_stable:
                return V, policy

            old_policy = policy



