from src.envs.gridworld_mdp import GridWorld
from src.iterative_policy_evaluation import IterativePolicyEvaluation
from src.policy.random_policy import EquiprobableRandomPolicy

mdp = GridWorld()
mdp.reset()
mdp._render("human")
policy = EquiprobableRandomPolicy()
policy_evaluation = IterativePolicyEvaluation(policy=policy, mdp=mdp)

policy_evaluation.estimate_state_value_function(theta=0.000000000000001)

