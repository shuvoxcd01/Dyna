from src.envs.gridworld_mdp import GridWorld
from src.policy.random_policy import EquiprobableRandomPolicy
from src.policy_evaluation.exact_policy_evaluation import ExactPolicyEvaluation
from src.policy_iteration.policy_iteration import PolicyIteration


mdp = GridWorld()
mdp.gamma = 0.999
policy_evaluator = ExactPolicyEvaluation(mdp=mdp)
initial_policy = EquiprobableRandomPolicy(mdp=mdp)

ob = PolicyIteration(mdp=mdp, policy_evaluator=policy_evaluator)
print(ob.do_policy_iteration(initial_policy=initial_policy))