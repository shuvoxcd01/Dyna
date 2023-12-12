from src.envs.mdp import MDP
from src.envs.gridworld import GridworldEnv


class GridWorld(MDP, GridworldEnv):
    def __init__(self, shape=[4,4]):
        MDP.__init__(self)
        GridworldEnv.__init__(self,shape=shape)
        self.states = list(set(self.P.keys()))
        self.actions = list(set(self.P[self.states[0]].keys()))
        self.gamma = 1.0
        self.reward_fn = self.get_reward
        self.transition_fn = self.get_transition_probability

    def get_transition_probability(self, next_state, reward, cur_state, action):
        prob, _next_state, _reward, done = self.P[cur_state][action][0]
        
        if next_state == _next_state and reward == _reward:
            return prob
        
        return 0.0

    def get_reward(self, cur_state, action):
        assert cur_state in self.states
        assert action in self.actions

        prob, _next_state, reward, done = self.P[cur_state][action][0]

        return reward
