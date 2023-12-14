from src.envs.gridworld_mdp import GridWorld
from src.exact_policy_evaluation import ExactPolicyEvaluation
from src.policy.random_policy import EquiprobableRandomPolicy

mdp = GridWorld()
mdp.reset()
mdp._render("human")
policy = EquiprobableRandomPolicy()
policy_evaluation = ExactPolicyEvaluation(policy=policy, mdp=mdp)
policy_evaluation.estimate_state_value_function_with_system_of_linear_equations()