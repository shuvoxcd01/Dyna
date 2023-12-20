from src.envs.gridworld_mdp import GridWorld
from src.policy.random_policy import EquiprobableRandomPolicy
from src.policy_evaluation.exact_policy_evaluation import ExactPolicyEvaluation

mdp = GridWorld()
mdp.reset()
mdp._render("human")
policy = EquiprobableRandomPolicy(mdp=mdp)
policy_evaluation = ExactPolicyEvaluation(mdp=mdp, policy=policy)
policy_evaluation.get_state_value_function_with_system_of_linear_equations()
