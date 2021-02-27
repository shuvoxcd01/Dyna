from src.gridworld_mdp import GridWorld
from src.iterative_policy_evaluation import IterativePolicyEvaluation
from src.random_policy import EquiprobableRandomPolicy

mdp = GridWorld()
policy = EquiprobableRandomPolicy()
policy_evaluation = IterativePolicyEvaluation(policy=policy, mdp=mdp)

policy_evaluation.estimate_state_value_function_inplace(theta=0.00000000001)

value_fn = policy_evaluation.get_value_fn()

states = list(mdp.states)
