# Policy-Evaluation
Iterative Policy Evaluation for estimating state-value function from an arbitrary policy.

# Environment
[Denny Britz's reinforcement-learning repository](https://github.com/dennybritz/reinforcement-learning.git) has been a great help in creating the environment. Most of the environment related code is taken from there.

Grid World environment from Sutton's Reinforcement Learning book chapter 4. You are an agent on an MxN grid and your goal is to reach the terminal state at the top left or the bottom right corner.

For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

x is your position and T are the two terminal states.

You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
Actions going off the edge leave you in your current state.
You receive a reward of -1 at each step until you reach a terminal state.

# Evaluation
Two types of policy evaluation implementations have been added.  
1. Exact Policy Evaluation: Uses Bellman's equation and solution to a system of linear equations.
2. Iterative Policy Evaluation: Evaluates policy in an iterative manner.

The Iterative Policy Evaluation offers two functions that perform policy evaluation. 
1. estimate_state_value_function_inplace
2. estimate_state_value_function

For the Grid World environment described above and a uniform random policy, all the functions converge to the following value assignment - which matches with Sutton & Barto's book (Reinforcement Learning An Introduction (Second Edition) See: Figure 4.1, Page: 77)

      0.0  -14.0  -20.0  -22.0   
    -14.0  -18.0  -20.0  -20.0   
    -20.0  -20.0  -18.0  -14.0   
    -22.0  -20.0  -14.0    0.0   
